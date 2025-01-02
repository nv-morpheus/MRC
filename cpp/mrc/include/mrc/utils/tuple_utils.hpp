/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstddef>
#include <tuple>

namespace mrc::utils {

template <typename TupleT, std::size_t... Is>
auto tuple_surely(TupleT&& tuple, std::index_sequence<Is...> /*unused*/)
{
    return std::tuple<typename std::tuple_element_t<Is, std::decay_t<TupleT>>::value_type...>(
        (std::get<Is>(tuple).value())...);
}

/**
 * @brief Converts a std::tuple<std::optional<T1>, std::optional<T2>, ...> to std::tuple<T1, T1, ...>
 *
 * @tparam TupleT The type of tuple
 * @param tuple
 * @return auto A new Tuple with `std::optional` types removed
 */
template <typename TupleT>
auto tuple_surely(TupleT&& tuple)
{
    return tuple_surely(std::forward<TupleT>(tuple),
                        std::make_index_sequence<std::tuple_size<std::decay_t<TupleT>>::value>());
}

template <typename TupleT, typename FuncT, std::size_t... Is>
void tuple_for_each(TupleT&& tuple, FuncT&& f, std::index_sequence<Is...> /*unused*/)
{
    (f(std::get<Is>(std::forward<TupleT>(tuple)), Is), ...);
}

/**
 * @brief Executes a function for each element of a tuple.
 *
 * @tparam TupleT The type of the tuple
 * @tparam FuncT The type of the lambda
 * @param tuple Tuple to run the function on
 * @param f A function which accepts an element of the tuple as the first arg and the index for the second arg.
 * Recommended to use `auto` or a templated lambda as the first argument
 */
template <typename TupleT, typename FuncT>
void tuple_for_each(TupleT&& tuple, FuncT&& f)
{
    tuple_for_each(std::forward<TupleT>(tuple),
                   std::forward<FuncT>(f),
                   std::make_index_sequence<std::tuple_size<std::decay_t<TupleT>>::value>());
}

// NOLINTBEGIN(readability-identifier-naming)
// Disable naming conventions for std library-like functions

/**
 * @brief Creates a tuple of N elements of type T. For example, `repeat_tuple_type<int, 3>` would be `std::tuple<int,
 * int, int>`
 *
 * @tparam T The type of the tuple
 * @tparam N The number of elements in the tuple
 */
template <typename T, size_t N>
class repeat_tuple_type
{
    template <typename = std::make_index_sequence<N>>
    struct impl;

    template <size_t... Is>
    struct impl<std::index_sequence<Is...>>
    {
        template <size_t>
        using wrap = T;

        using type = std::tuple<wrap<Is>...>;
    };

  public:
    using type = typename impl<>::type;
};

template <typename T, size_t N>
using repeat_tuple_type_t = typename repeat_tuple_type<T, N>::type;

// NOLINTEND(readability-identifier-naming)

}  // namespace mrc::utils
