/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"

#include <type_traits>

namespace mrc::node {

// Ignore naming conventions here to match <type_traits> (Available after clang-tidy-14)
// NOLINTBEGIN(readability-identifier-naming)

template <bool T>
struct bool_value  // NOLINT(readability-identifier-naming
{
    static constexpr bool value = T;  // NOLINT(readability-identifier-naming)

    constexpr operator bool() const noexcept
    {
        return value;
    }

    constexpr bool operator()() const noexcept
    {
        return value;
    }
};

template <template <typename...> class BaseT, typename DerivedT>
struct is_base_of_template_impl  // NOLINT(readability-identifier-naming)
{
    /*
      Note: As of c++17, std::is_base_of is not sufficient to test for specialized base classes.
      Declare a function 'test', with two signature patterns, one for a class of BaseT, and one for anything else. This
      allows for the subsequent decltype(test(std::declval<DerivedT*>)) pattern, which will return std::true_type if
      DerivedT has a type that can be converted to BaseT, std::false_type otherwise.
    */
    template <typename... ArgsT>
    static constexpr std::true_type test(const BaseT<ArgsT...>*);
    static constexpr std::false_type test(...);

    using type = decltype(test(std::declval<DerivedT*>()));  // NOLINT(readability-identifier-naming)
};

/**
 * @brief Determines if a type DerivedT is derived from BaseT when both types have template arguments.
 */
template <template <typename...> class BaseT, typename DerivedT>
// NOLINTNEXTLINE(readability-identifier-naming)
using is_base_of_template = typename is_base_of_template_impl<BaseT, DerivedT>::type;

template <typename T, typename = void>
struct is_sink : bool_value<false>  // NOLINT(readability-identifier-naming)
{};

/**
 * @brief Indicates if a type, T, is a Segment Sink. Can be used as both a test (using `value`) and to derive the Sink
 * type (using `type`)
 *
 * @tparam T Type to test
 */
template <typename T>
struct is_sink<T, std::enable_if_t<is_base_of_template<node::SinkProperties, T>::value>>
  : bool_value<true>  // NOLINT(readability-identifier-naming)
{
  private:
    template <typename... ArgsT>
    static constexpr typename node::SinkProperties<ArgsT...>::sink_type_t test(const node::SinkProperties<ArgsT...>*);

  public:
    using type = decltype(test(std::declval<T*>()));  // NOLINT(readability-identifier-naming)
};

template <typename T, typename = void>
struct is_source : bool_value<false>  // NOLINT(readability-identifier-naming)
{};

/**
 * @brief Indicates if a type, T, is a Segment Source. Can be used as both a test (using `value`) and to derive the
 * Source type (using `type`)
 *
 * @tparam T Type to test
 */
template <typename T>
struct is_source<T, std::enable_if_t<is_base_of_template<SourceProperties, T>::value>>
  : bool_value<true>  // NOLINT(readability-identifier-naming)
{
  private:
    template <typename... ArgsT>
    static constexpr typename SourceProperties<ArgsT...>::source_type_t test(const SourceProperties<ArgsT...>*);

  public:
    using type = decltype(test(std::declval<T*>()));  // NOLINT(readability-identifier-naming)
};

template <typename T, typename = void>
struct is_node : bool_value<false>  // NOLINT(readability-identifier-naming)
{};

/**
 * @brief Indicates if a type, T, is both a Segment Sink and a Segment Source, also known as a Segment Node. Can be used
 * as both a test (using `value`) and to derive the Sink type (using `sink_type`) and Source type (using `source_type`)
 *
 * @tparam T
 */
template <typename T>
struct is_node<T, std::enable_if_t<is_sink<T>{} && is_source<T>{}>>
  : bool_value<true>  // NOLINT(readability-identifier-naming)
{
  public:
    using sink_type   = typename is_sink<T>::type;    // NOLINT(readability-identifier-naming)
    using source_type = typename is_source<T>::type;  // NOLINT(readability-identifier-naming)
};

// NOLINTEND(readability-identifier-naming)
}  // namespace mrc::node
