/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/node/node_parent.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/tuple_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/unbuffered_channel.hpp>
#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

// IWYU pragma: begin_exports

namespace mrc::node {

class ZipTypelessBase
{
  public:
    virtual ~ZipTypelessBase() = default;
};

template <typename... TypesT>
class ZipBase
{};

template <typename... InputT, typename OutputT>
class ZipBase<std::tuple<InputT...>, OutputT> : public ZipTypelessBase,
                                                public WritableAcceptor<OutputT>,
                                                public HeterogeneousNodeParent<edge::IWritableProvider<InputT>...>
{
  public:
    using input_tuple_t = std::tuple<InputT...>;
    using output_t      = OutputT;

  private:
    template <typename T>
    using queue_t = BufferedChannel<T>;
    template <typename T>
    using wrapped_queue_t   = std::unique_ptr<queue_t<T>>;
    using queues_tuple_type = std::tuple<wrapped_queue_t<InputT>...>;

    template <std::size_t... Is>
    static auto build_ingress(ZipBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

    static auto build_queues(size_t channel_size)
    {
        return std::make_tuple(std::make_unique<queue_t<InputT>>(channel_size)...);
    }

    template <std::size_t... Is>
    static std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...>
    build_child_pairs(ZipBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(
            std::make_pair(MRC_CONCAT_STR("sink[" << Is << "]"), std::ref(*self->get_sink<Is>()))...);
    }

    template <std::size_t I = 0>
    channel::Status tuple_pop_each(queues_tuple_type& queues_tuple, input_tuple_t& output_tuple)
    {
        channel::Status status = std::get<I>(queues_tuple)->await_read(std::get<I>(output_tuple));

        if constexpr (I + 1 < sizeof...(InputT))
        {
            // Iterate to the next index
            channel::Status inner_status = tuple_pop_each<I + 1>(queues_tuple, output_tuple);

            // If the inner status failed, return that, otherwise return our status
            status = inner_status == channel::Status::success ? status : inner_status;
        }

        return status;
    }

  public:
    ZipBase(size_t max_outstanding = channel::default_channel_size()) :
      m_queues(build_queues(max_outstanding)),
      m_upstream_holders(build_ingress(const_cast<ZipBase*>(this), std::index_sequence_for<InputT...>{}))
    {
        // Must be sure to set any array values
        m_queue_counts.fill(0);
    }

    ~ZipBase() override = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, InputT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

    std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...> get_children_refs()
        const override
    {
        return build_child_pairs(const_cast<ZipBase*>(this), std::index_sequence_for<InputT...>{});
    }

  protected:
    template <size_t N>
    class Upstream : public ForwardingWritableProvider<NthTypeOf<N, InputT...>>
    {
        using upstream_t = NthTypeOf<N, InputT...>;

      public:
        Upstream(ZipBase& parent) : m_parent(parent) {}

      protected:
        channel::Status on_next(upstream_t&& data) override
        {
            return m_parent.upstream_await_write<N>(std::move(data));
        }

        void on_complete() override
        {
            m_parent.edge_complete<N>();
        }

      private:
        ZipBase& m_parent;
    };

  private:
    template <size_t N>
    channel::Status upstream_await_write(NthTypeOf<N, InputT...> value)
    {
        // Push before locking so we dont deadlock
        auto push_status = std::get<N>(m_queues)->await_write(std::move(value));

        if (push_status != channel::Status::success)
        {
            return push_status;
        }

        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the counts array
        m_queue_counts[N]++;

        if (m_queue_counts[N] == m_max_queue_count)
        {
            // Close the queue to prevent pushing more messages
            std::get<N>(m_queues)->close_channel();
        }

        DCHECK_LE(m_queue_counts[N], m_max_queue_count) << "Queue count has surpassed the max count";

        // See if we have values in every queue
        auto all_queues_have_value = std::transform_reduce(m_queue_counts.begin(),
                                                           m_queue_counts.end(),
                                                           true,
                                                           std::logical_and<>(),
                                                           [this](const size_t& v) {
                                                               return v > m_pull_count;
                                                           });

        channel::Status status = channel::Status::success;

        if (all_queues_have_value)
        {
            // For each tuple, pop a value off
            std::tuple<InputT...> new_val;

            auto channel_status = tuple_pop_each(m_queues, new_val);

            DCHECK_EQ(channel_status, channel::Status::success) << "Queues returned failed status";

            // Push the new value
            status = this->get_writable_edge()->await_write(this->convert_value(std::move(new_val)));

            m_pull_count++;
        }

        return status;
    }

    template <size_t N>
    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        if (m_queue_counts[N] < m_max_queue_count)
        {
            // We are setting a new lower limit. Check to make sure this isnt an issue
            m_max_queue_count = m_queue_counts[N];

            utils::tuple_for_each(m_queues,
                                  [this]<typename QueueValueT>(std::unique_ptr<queue_t<QueueValueT>>& q, size_t idx) {
                                      if (m_queue_counts[idx] >= m_max_queue_count)
                                      {
                                          // Close the channel
                                          q->close_channel();

                                          if (m_queue_counts[idx] > m_max_queue_count)
                                          {
                                              LOG(ERROR)
                                                  << "Unbalanced count in upstream sources for Zip operator. Upstream '"
                                                  << N << "' ended with " << m_queue_counts[N] << " elements but "
                                                  << m_queue_counts[idx]
                                                  << " elements have already been pushed by upstream '" << idx << "'";
                                          }
                                      }
                                  });
        }

        m_completions++;

        if (m_completions == sizeof...(InputT))
        {
            // Warn on any left over values
            auto left_over_messages = std::transform_reduce(m_queue_counts.begin(),
                                                            m_queue_counts.end(),
                                                            0,
                                                            std::plus<>(),
                                                            [this](const size_t& v) {
                                                                return v - m_pull_count;
                                                            });
            if (left_over_messages > 0)
            {
                LOG(ERROR) << "Unbalanced count in upstream sources for Zip operator. " << left_over_messages
                           << " messages were left in the queues";
            }

            // Finally, drain the queues of any remaining values
            utils::tuple_for_each(m_queues,
                                  []<typename QueueValueT>(std::unique_ptr<queue_t<QueueValueT>>& q, size_t idx) {
                                      QueueValueT value;

                                      while (q->await_read(value) == channel::Status::success) {}
                                  });

            WritableAcceptor<output_t>::release_edge_connection();
        }
    }

    virtual output_t convert_value(input_tuple_t&& data) = 0;

    mutable Mutex m_mutex;

    // Once an upstream is closed, this is set representing the max number of values in a queue before its closed
    size_t m_max_queue_count{std::numeric_limits<size_t>::max()};

    // Counts the number of upstream completions. When m_completions == sizeof...(TypesT), the downstream edges are
    // released
    size_t m_completions{0};

    // Holds the number of values pushed to each queue
    std::array<size_t, sizeof...(InputT)> m_queue_counts;

    // The number of messages pulled off the queue
    size_t m_pull_count{0};

    // Queue used to allow backpressure to upstreams
    queues_tuple_type m_queues;

    // Upstream edges
    std::tuple<std::shared_ptr<WritableProvider<InputT>>...> m_upstream_holders;
};

template <typename...>
class ZipComponent;

template <typename... InputT, typename OutputT>
class ZipComponent<std::tuple<InputT...>, OutputT> : public ZipBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t        = ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;
};

// Specialization for Zip with a default output type
template <typename... InputT>
class ZipComponent<std::tuple<InputT...>> : public ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>
{
  public:
    using base_t        = ZipBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;

  private:
    output_t convert_value(input_tuple_t&& data) override
    {
        // No change to the output type
        return std::move(data);
    }
};

template <typename...>
class ZipTransformComponent;

template <typename... InputT, typename OutputT>
class ZipTransformComponent<std::tuple<InputT...>, OutputT> : public ZipBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t         = ZipBase<std::tuple<InputT...>, OutputT>;
    using input_tuple_t  = typename base_t::input_tuple_t;
    using output_t       = typename base_t::output_t;
    using transform_fn_t = std::function<output_t(input_tuple_t&&)>;

    ZipTransformComponent(transform_fn_t transform_fn, size_t max_outstanding = 64) :
      base_t(max_outstanding),
      m_transform_fn(std::move(transform_fn))
    {}

  private:
    output_t convert_value(input_tuple_t&& data) override
    {
        return m_transform_fn(std::move(data));
    }

    transform_fn_t m_transform_fn;
};

}  // namespace mrc::node

// IWYU pragma: end_exports
