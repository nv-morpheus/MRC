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

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/unbuffered_channel.hpp>

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mrc::node {

template <std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type for_each(std::tuple<Tp...>&, FuncT)  // Unused arguments
                                                                                                    // are given no
                                                                                                    // names.
{}

template <std::size_t I = 0, typename FuncT, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), void>::type for_each(std::tuple<Tp...>& t, FuncT f)
{
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
}

template <typename TupleT, typename FuncT, std::size_t... Is>
auto tuple_for_each(const TupleT& tuple, FuncT f, std::index_sequence<Is...>)
{
    std::tuple<typename std::invoke_result_t<FuncT, std::tuple_element_t<Is, std::decay_t<TupleT>>>...> output;

    return std::tuple<typename std::invoke_result_t<FuncT, std::tuple_element_t<Is, std::decay_t<TupleT>>>...>(
        (f(std::get<Is>(std::forward<TupleT>(tuple))))...);
}

template <typename TupleT, typename FuncT, std::size_t... Is>
auto tuple_for_each(const TupleT& tuple, FuncT f, std::index_sequence<Is...>)
{
    return std::tuple<typename std::invoke_result_t<FuncT, std::tuple_element_t<Is, std::decay_t<TupleT>>>...>(
        (f(std::get<Is>(std::forward<TupleT>(tuple))))...);
}

template <typename TupleT, typename FuncT>
auto tuple_for_each(TupleT&& tuple, FuncT&& f)
{
    return tuple_for_each(std::forward<TupleT>(tuple),
                          std::forward<FuncT>(f),
                          std::make_index_sequence<std::tuple_size<std::decay_t<TupleT>>::value>());
}

template <typename T, std::size_t... Is>
bool array_reduce_ge_zero(const std::array<T, sizeof...(Is)>& array, std::index_sequence<Is...>)
{
    return ((array[Is] > 0) && ...);
}

// Retur
template <typename ArrayT>
bool array_reduce_ge_zero(const ArrayT& array)
{
    return array_reduce_ge_zero(array, std::make_index_sequence<std::tuple_size<std::decay_t<ArrayT>>::value>());
}

template <typename... TypesT>
class Zip : public WritableAcceptor<std::tuple<TypesT...>>
{
    template <std::size_t... Is>
    static auto build_ingress(Zip* self, std::index_sequence<Is...> /*unused*/)
    {
        return std::make_tuple(std::make_shared<Upstream<Is>>(*self)...);
    }

    static auto build_vectors()
    {
        return std::make_tuple(std::vector<TypesT>()...);
    }

    static auto build_queues()
    {
        return std::make_tuple(std::make_shared<boost::fibers::buffered_channel<TypesT>>(128)...);
        // return std::make_tuple(boost::fibers::unbuffered_channel<TypesT>()...);
        // return std::tuple<boost::fibers::unbuffered_channel<int>,
        // boost::fibers::unbuffered_channel<float>>(boost::fibers::unbuffered_channel<int>(),
        // boost::fibers::unbuffered_channel<float>());
        // return std::make_tuple(QueueBuilder<Is>::build()...);
    }

  public:
    Zip() :
      m_queues(build_queues()),
      m_vectors(build_vectors()),
      m_upstream_holders(build_ingress(const_cast<Zip*>(this), std::index_sequence_for<TypesT...>{}))
    {}

    virtual ~Zip() = default;

    template <size_t N>
    std::shared_ptr<edge::IWritableProvider<NthTypeOf<N, TypesT...>>> get_sink() const
    {
        return std::get<N>(m_upstream_holders);
    }

  protected:
    template <size_t N>
    struct QueueBuilder : public WritableProvider<NthTypeOf<N, TypesT...>>
    {
        using upstream_t = NthTypeOf<N, TypesT...>;

        static auto build()
        {
            return boost::fibers::unbuffered_channel<upstream_t>();
        }
    };

    template <size_t N>
    class Upstream : public WritableProvider<NthTypeOf<N, TypesT...>>
    {
        using upstream_t = NthTypeOf<N, TypesT...>;

      public:
        Upstream(Zip& parent)
        {
            this->init_owned_edge(std::make_shared<InnerEdge>(parent));
        }

      private:
        class InnerEdge : public edge::IEdgeWritable<NthTypeOf<N, TypesT...>>
        {
          public:
            InnerEdge(Zip& parent) : m_parent(parent) {}
            ~InnerEdge()
            {
                m_parent.edge_complete();
            }

            virtual channel::Status await_write(upstream_t&& data)
            {
                return m_parent.upstream_await_write<N>(std::move(data));
            }

          private:
            Zip& m_parent;
        };
    };

  private:
    template <size_t N>
    channel::Status upstream_await_write(NthTypeOf<N, TypesT...> value)
    {
        // Push before locking so we dont deadlock
        std::get<N>(m_queues)->push(std::move(value));

        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Update the counts array
        m_queue_counts[N]++;

        // See if we have values in every queue
        auto all_queues_have_value = std::transform_reduce(m_queue_counts.begin(),
                                                           m_queue_counts.end(),
                                                           true,
                                                           std::logical_and<>(),
                                                           [](const size_t& v) {
                                                               return v > 0;
                                                           });

        channel::Status status = channel::Status::success;

        if (all_queues_have_value)
        {
            // For each tuple, pop a value off
            // std::tuple<TypesT...> new_val = tuple_for_each(m_queues, [](const auto& q){

            // });
            std::tuple<TypesT...> new_val;

            // Reduce the counts by 1
            for (auto& c : m_queue_counts)
            {
                c--;
            }

            // Push the new value
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
            // Warn on any left over values

            WritableAcceptor<std::tuple<TypesT...>>::release_edge_connection();
        }
    }

    boost::fibers::mutex m_mutex;
    size_t m_values_set{0};
    size_t m_completions{0};
    std::array<size_t, sizeof...(TypesT)> m_queue_counts;
    std::tuple<std::shared_ptr<boost::fibers::buffered_channel<TypesT>>...> m_queues;
    std::tuple<std::vector<TypesT>...> m_vectors;

    std::tuple<std::shared_ptr<WritableProvider<TypesT>>...> m_upstream_holders;
};

std::tuple<boost::fibers::unbuffered_channel<int>, boost::fibers::unbuffered_channel<float>> test2(
    boost::fibers::unbuffered_channel<int>(),
    boost::fibers::unbuffered_channel<float>());

Zip<int, float> test;

}  // namespace mrc::node
