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

#include "mrc/channel/status.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/type_utils.hpp"

#include <boost/fiber/mutex.hpp>

#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace mrc::node {

template <typename StdTuple, std::size_t... Is>
auto surely(const StdTuple& stdTuple, std::index_sequence<Is...>)
{
    return std::tuple<typename std::tuple_element_t<Is, std::decay_t<StdTuple>>::value_type...>(
        (std::get<Is>(stdTuple).value())...);
}

// Converts a std::tuple<std::optional<T1>, std::optional<T2>, ...> to std::tuple<T1, T1, ...>
template <typename StdTuple>
auto surely(const StdTuple& stdTuple)
{
    return surely(stdTuple, std::make_index_sequence<std::tuple_size<std::decay_t<StdTuple>>::value>());
}

template <typename... TypesT>
class CombineLatest : public WritableAcceptor<std::tuple<TypesT...>>
{
    template <std::size_t... Is>
    static auto build_ingress(CombineLatest* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

  public:
    CombineLatest() :
      m_upstream_holders(build_ingress(const_cast<CombineLatest*>(this), std::index_sequence_for<TypesT...>{}))
    {}

    virtual ~CombineLatest() = default;

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
        Upstream(CombineLatest& parent)
        {
            this->init_owned_edge(std::make_shared<InnerEdge>(parent));
        }

      private:
        class InnerEdge : public edge::IEdgeWritable<NthTypeOf<N, TypesT...>>
        {
          public:
            InnerEdge(CombineLatest& parent) : m_parent(parent) {}
            ~InnerEdge()
            {
                m_parent.edge_complete();
            }

            virtual channel::Status await_write(upstream_t&& data)
            {
                return m_parent.set_upstream_value<N>(std::move(data));
            }

          private:
            CombineLatest& m_parent;
        };
    };

  private:
    template <size_t N>
    channel::Status set_upstream_value(NthTypeOf<N, TypesT...> value)
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the current value
        auto& nth_val = std::get<N>(m_state);

        if (!nth_val.has_value())
        {
            ++m_values_set;
        }

        nth_val = std::move(value);

        channel::Status status = channel::Status::success;

        // Check if we should push the new value
        if (m_values_set == sizeof...(TypesT))
        {
            std::tuple<TypesT...> new_val = surely(m_state);

            status = this->get_writable_edge()->await_write(std::move(new_val));
        }

        return status;
    }

    void edge_complete()
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        m_completions++;

        if (m_completions == sizeof...(TypesT))
        {
            // Clear the held tuple to remove any dangling values
            m_state = std::tuple<std::optional<TypesT>...>();

            WritableAcceptor<std::tuple<TypesT...>>::release_edge_connection();
        }
    }

    boost::fibers::mutex m_mutex;
    size_t m_values_set{0};
    size_t m_completions{0};
    std::tuple<std::optional<TypesT>...> m_state;

    std::tuple<std::shared_ptr<WritableProvider<TypesT>>...> m_upstream_holders;
};

}  // namespace mrc::node
