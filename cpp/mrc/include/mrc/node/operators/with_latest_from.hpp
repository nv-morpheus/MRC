/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/node/node_parent.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/tuple_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/mutex.hpp>
#include <glog/logging.h>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace mrc::node {

class WithLatestFromTypelessBase
{
  public:
    virtual ~WithLatestFromTypelessBase() = default;
};

template <typename... TypesT>
class WithLatestFromBase
{};

template <typename... InputT, typename OutputT>
class WithLatestFromBase<std::tuple<InputT...>, OutputT>
  : public WithLatestFromTypelessBase,
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
    static auto build_ingress(WithLatestFromBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

    static auto build_queues(size_t channel_size)
    {
        return std::make_tuple(std::make_unique<queue_t<InputT>>(channel_size)...);
    }

    template <std::size_t... Is>
    static std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...>
    build_child_pairs(WithLatestFromBase* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(
            std::make_pair(MRC_CONCAT_STR("sink[" << Is << "]"), std::ref(*self->get_sink<Is>()))...);
    }

  public:
    WithLatestFromBase(size_t max_outstanding = channel::default_channel_size()) :
      m_primary_queue(std::make_unique<queue_t<NthTypeOf<0, InputT...>>>(max_outstanding)),
      m_upstream_holders(build_ingress(const_cast<WithLatestFromBase*>(this), std::index_sequence_for<InputT...>{}))
    {}

    virtual ~WithLatestFromBase() = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, InputT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

    std::tuple<std::pair<std::string, std::reference_wrapper<edge::IWritableProvider<InputT>>>...> get_children_refs()
        const override
    {
        return build_child_pairs(const_cast<WithLatestFromBase*>(this), std::index_sequence_for<InputT...>{});
    }

  protected:
    template <size_t N>
    class Upstream : public ForwardingWritableProvider<NthTypeOf<N, InputT...>>
    {
        using upstream_t = NthTypeOf<N, InputT...>;

      public:
        Upstream(WithLatestFromBase& parent) : m_parent(parent) {}

      protected:
        channel::Status on_next(upstream_t&& data) override
        {
            return m_parent.upstream_await_write<N>(std::move(data));
        }

        void on_complete() override
        {
            m_parent.edge_complete();
        }

      private:
        WithLatestFromBase& m_parent;
    };

  private:
    template <size_t N>
    channel::Status upstream_await_write(NthTypeOf<N, InputT...> value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Get a reference to the current value
        auto& nth_val = std::get<N>(m_state);

        // Check if we have fully initialized
        if (m_values_set < sizeof...(InputT))
        {
            if (!nth_val.has_value())
            {
                ++m_values_set;
            }

            // Move the value into the state
            nth_val = std::move(value);

            // For the primary upstream only, move the value onto a queue
            if constexpr (N == 0)
            {
                // Temporarily unlock to prevent deadlock
                lock.unlock();

                Unwinder relock([&]() {
                    lock.lock();
                });

                // Move it into the queue
                CHECK_EQ(m_primary_queue->await_write(std::move(nth_val.value())), channel::Status::success);
            }

            // Check if this put us over the edge
            if (m_values_set == sizeof...(InputT))
            {
                // Need to complete initialization. First close the primary channel
                m_primary_queue->close_channel();

                auto& primary_val = std::get<0>(m_state);

                // Loop over the values in the queue, pushing each one
                while (m_primary_queue->await_read(primary_val.value()) == channel::Status::success)
                {
                    std::tuple<InputT...> new_val = utils::tuple_surely(m_state);

                    CHECK_EQ(this->get_writable_edge()->await_write(this->convert_value(std::move(new_val))),
                             channel::Status::success);
                }
            }
        }
        else
        {
            // Move the value into the state
            nth_val = std::move(value);

            // Only when we are the primary, do we push a new value
            if constexpr (N == 0)
            {
                std::tuple<InputT...> new_val = utils::tuple_surely(m_state);

                return this->get_writable_edge()->await_write(this->convert_value(std::move(new_val)));
            }
        }

        return channel::Status::success;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(InputT))
        {
            NthTypeOf<0, InputT...> tmp;
            bool had_values = false;

            // Try to clear out any values left in the channel
            while (m_primary_queue->await_read(tmp) == channel::Status::success)
            {
                had_values = true;
            }

            LOG_IF(ERROR, had_values) << "The primary source values were never pushed downstream. Ensure all upstream "
                                         "sources pushed at least 1 value";

            // Clear the held tuple to remove any dangling values
            m_state = std::tuple<std::optional<InputT>...>();

            WritableAcceptor<OutputT>::release_edge_connection();
        }
    }

    virtual output_t convert_value(input_tuple_t&& data) = 0;

    mutable Mutex m_mutex;

    // The number of elements that have been set. Can start emitting when m_values_set == sizeof...(TypesT)
    size_t m_values_set{0};

    // Counts the number of upstream completions. When m_completions == sizeof...(TypesT), the downstream edges are
    // released
    size_t m_completions{0};

    // Holds onto the latest values to eventually push when new ones are emitted
    std::tuple<std::optional<InputT>...> m_state;

    // Queue to allow backpressure to upstreams. Only 1 queue for the primary is needed
    wrapped_queue_t<NthTypeOf<0, InputT...>> m_primary_queue;

    // Upstream edges
    std::tuple<std::shared_ptr<WritableProvider<InputT>>...> m_upstream_holders;
};

template <typename...>
class WithLatestFromComponent;

template <typename... InputT, typename OutputT>
class WithLatestFromComponent<std::tuple<InputT...>, OutputT>
  : public WithLatestFromBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t        = WithLatestFromBase<std::tuple<InputT...>, std::tuple<InputT...>>;
    using input_tuple_t = typename base_t::input_tuple_t;
    using output_t      = typename base_t::output_t;
};

// Specialization for WithLatestFromBase with a default output type
template <typename... InputT>
class WithLatestFromComponent<std::tuple<InputT...>>
  : public WithLatestFromBase<std::tuple<InputT...>, std::tuple<InputT...>>
{
  public:
    using base_t        = WithLatestFromBase<std::tuple<InputT...>, std::tuple<InputT...>>;
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
class WithLatestFromTransformComponent;

template <typename... InputT, typename OutputT>
class WithLatestFromTransformComponent<std::tuple<InputT...>, OutputT>
  : public WithLatestFromBase<std::tuple<InputT...>, OutputT>
{
  public:
    using base_t         = WithLatestFromBase<std::tuple<InputT...>, OutputT>;
    using input_tuple_t  = typename base_t::input_tuple_t;
    using output_t       = typename base_t::output_t;
    using transform_fn_t = std::function<output_t(input_tuple_t&&)>;

    WithLatestFromTransformComponent(transform_fn_t transform_fn, size_t max_outstanding = 64) :
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
