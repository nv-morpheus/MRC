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
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/tuple_utils.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/mutex.hpp>
#include <glog/logging.h>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace mrc::node {

template <typename... TypesT>
class WithLatestFrom : public WritableAcceptor<std::tuple<TypesT...>>
{
    template <typename T>
    using queue_t = BufferedChannel<T>;
    template <typename T>
    using wrapped_queue_t = std::unique_ptr<queue_t<T>>;
    using output_t        = std::tuple<TypesT...>;

    template <std::size_t... Is>
    static auto build_ingress(WithLatestFrom* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

  public:
    WithLatestFrom() :
      m_primary_queue(std::make_unique<queue_t<NthTypeOf<0, TypesT...>>>()),
      m_upstream_holders(build_ingress(const_cast<WithLatestFrom*>(this), std::index_sequence_for<TypesT...>{}))
    {}

    virtual ~WithLatestFrom() = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, TypesT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

  protected:
    template <size_t N>
    class Upstream : public WritableProvider<NthTypeOf<N, TypesT...>>
    {
        using upstream_t = NthTypeOf<N, TypesT...>;

      public:
        Upstream(WithLatestFrom& parent)
        {
            this->init_owned_edge(std::make_shared<InnerEdge>(parent));
        }

      private:
        class InnerEdge : public edge::IEdgeWritable<NthTypeOf<N, TypesT...>>
        {
          public:
            InnerEdge(WithLatestFrom& parent) : m_parent(parent) {}
            ~InnerEdge()
            {
                m_parent.edge_complete();
            }

            virtual channel::Status await_write(upstream_t&& data)
            {
                return m_parent.set_upstream_value<N>(std::move(data));
            }

          private:
            WithLatestFrom& m_parent;
        };
    };

  private:
    template <size_t N>
    channel::Status set_upstream_value(NthTypeOf<N, TypesT...> value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Get a reference to the current value
        auto& nth_val = std::get<N>(m_state);

        // Check if we have fully initialized
        if (m_values_set < sizeof...(TypesT))
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
            if (m_values_set == sizeof...(TypesT))
            {
                // Need to complete initialization. First close the primary channel
                m_primary_queue->close_channel();

                auto& primary_val = std::get<0>(m_state);

                // Loop over the values in the queue, pushing each one
                while (m_primary_queue->await_read(primary_val.value()) == channel::Status::success)
                {
                    std::tuple<TypesT...> new_val = utils::tuple_surely(m_state);

                    CHECK_EQ(this->get_writable_edge()->await_write(std::move(new_val)), channel::Status::success);
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
                std::tuple<TypesT...> new_val = utils::tuple_surely(m_state);

                return this->get_writable_edge()->await_write(std::move(new_val));
            }
        }

        return channel::Status::success;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(TypesT))
        {
            NthTypeOf<0, TypesT...> tmp;
            bool had_values = false;

            // Try to clear out any values left in the channel
            while (m_primary_queue->await_read(tmp) == channel::Status::success)
            {
                had_values = true;
            }

            LOG_IF(ERROR, had_values) << "The primary source values were never pushed downstream. Ensure all upstream "
                                         "sources pushed at least 1 value";

            // Clear the held tuple to remove any dangling values
            m_state = std::tuple<std::optional<TypesT>...>();

            WritableAcceptor<std::tuple<TypesT...>>::release_edge_connection();
        }
    }

    boost::fibers::mutex m_mutex;

    // The number of elements that have been set. Can start emitting when m_values_set == sizeof...(TypesT)
    size_t m_values_set{0};

    // Counts the number of upstream completions. When m_completions == sizeof...(TypesT), the downstream edges are
    // released
    size_t m_completions{0};

    // Holds onto the latest values to eventually push when new ones are emitted
    std::tuple<std::optional<TypesT>...> m_state;

    // Queue to allow backpressure to upstreams. Only 1 queue for the primary is needed
    wrapped_queue_t<NthTypeOf<0, TypesT...>> m_primary_queue;

    // Upstream edges
    std::tuple<std::shared_ptr<WritableProvider<TypesT>>...> m_upstream_holders;
};

}  // namespace mrc::node
