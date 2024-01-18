/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <type_traits>

namespace mrc {

// template <typename ContainerT>
// struct is_sequence_container : std::false_type
// {};

// template <typename... T>
// struct is_sequence_container<std::array<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_sequence_container<std::vector<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_sequence_container<std::deque<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_sequence_container<std::forward_list<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_sequence_container<std::list<T...>> : std::true_type
// {};

// template <typename ContainerT>
// struct is_associative_container : std::false_type
// {};

// template <typename... T>
// struct is_associative_container<std::set<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::map<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::multiset<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::multimap<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::unordered_set<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::unordered_map<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::unordered_multiset<T...>> : std::true_type
// {};

// template <typename... T>
// struct is_associative_container<std::unordered_multimap<T...>> : std::true_type
// {};

// inline constexpr auto is_container_impl(...) -> std::false_type
// {
//     return std::false_type{};
// }

// template <typename C>
// constexpr auto is_container_impl(C const* c) -> decltype(begin(*c), end(*c), std::true_type{})
// {
//     return std::true_type{};
// }

// template <typename C>
// constexpr auto is_container(C const& c) -> decltype(is_container_impl(&c))
// {
//     return is_container_impl(&c);
// }

// inline constexpr auto is_associative_container_impl(...) -> std::false_type
// {
//     return std::false_type{};
// }

// template <typename C, typename = typename C::key_type>
// constexpr auto is_associative_container_impl(C const*) -> std::true_type
// {
//     return std::true_type{};
// }

// template <typename C>
// constexpr auto is_associative_container(C const& c) -> decltype(is_associative_container_impl(&c))
// {
//     return is_associative_container_impl(&c);
// }

template <typename... T>
struct is_container_helper
{};

template <typename T, typename = void>
struct is_container : std::false_type
{};

template <typename T>
struct is_container<T,
                    std::conditional_t<false,
                                       is_container_helper<typename T::value_type,
                                                           typename T::size_type,
                                                           typename T::iterator,
                                                           typename T::const_iterator,
                                                           decltype(std::declval<T>().size()),
                                                           decltype(std::declval<T>().begin()),
                                                           decltype(std::declval<T>().end()),
                                                           decltype(std::declval<T>().cbegin()),
                                                           decltype(std::declval<T>().cend())>,
                                       void>> : public std::true_type
{};

// template <typename T, typename = void>
// struct is_sequence_container : std::false_type
// {};

// template <typename T>
// struct is_sequence_container<T,
//                              std::conditional_t<false,
//                                                 is_container_helper<is_container<T>>,
//                                                 void>> : public std::true_type
// {};

template <typename T, typename = void>
struct is_associative_container : std::false_type
{};

// template <typename T>
// struct is_associative_container<
//     T,
//     std::conditional_t<false,
//                        is_container_helper<is_container<T>,
//                                            typename T::key_type,
//                                            decltype(std::declval<T>().find(std::declval<typename T::key_type>))>,
//                        void>> : public std::true_type
// {};

template <typename T>
struct is_associative_container<T,
                                std::conditional_t<false,
                                                   is_container_helper<typename T::value_type,
                                                                       typename T::size_type,
                                                                       typename T::iterator,
                                                                       typename T::const_iterator,
                                                                       typename T::key_type,
                                                                       decltype(std::declval<T>().size()),
                                                                       decltype(std::declval<T>().begin()),
                                                                       decltype(std::declval<T>().end()),
                                                                       decltype(std::declval<T>().cbegin()),
                                                                       decltype(std::declval<T>().cend())>,
                                                   void>> : public std::true_type
{};

}  // namespace mrc
