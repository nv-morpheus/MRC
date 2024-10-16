/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <type_traits>
#include <utility>

namespace mrc {

template <typename T>
struct is_shared_ptr : std::false_type
{};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_shared_ptr_v = is_shared_ptr<T>::value;  // NOLINT

template <typename T>
struct is_unique_ptr : std::false_type
{};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type
{};

template <typename T>
inline constexpr bool is_unique_ptr_v = is_unique_ptr<T>::value;  // NOLINT

template <typename TargetT, typename SourceT>
inline std::unique_ptr<TargetT> dynamic_pointer_cast(std::unique_ptr<SourceT>&& source)
{
    auto* const target = dynamic_cast<TargetT*>(source.get());
    if (target == nullptr)
    {
        return nullptr;
    }
    source.release();
    return std::unique_ptr<TargetT>(target);
}

template <typename T>
struct is_smart_ptr : std::integral_constant<bool, is_unique_ptr<T>::value or is_shared_ptr<T>::value>
{};

template <typename T>
inline constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;  // NOLINT
/*
template <typename T, typename = void>
struct is_valid_type : std::false_type
{};

template <typename T>
struct is_valid_type<T, std::enable_if_t<std::is_object_v<T> and not is_smart_ptr<T>>> : std::true_type
{};

template <typename T, typename = void>
struct use_value_semantics : std::false_type
{};

template <typename T>
struct use_value_semantics<T, std::enable_if_t<std::is_scalar_t<T> and is_valid_type<T>::value>> : std::true_type
{};

template <typename T, typename = void>
struct use_object_semantics : std::false_type
{};

template <typename T>
struct use_object_semantics<
    T,
    std::enable_if_t<std::is_class_v<T> and not use_value_semantics<T>::value and is_valid_type<T>::value>>
  : std::true_type
{};
*/

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

template <template <typename...> class BaseT, typename DerivedT>
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr bool is_base_of_template_v = is_base_of_template<BaseT, DerivedT>::value;

template <typename T, typename... TsT>
struct first_non_void_type  // NOLINT
{
    using type_t = std::conditional_t<std::is_same_v<T, void>, typename first_non_void_type<TsT...>::type_t, T>;
};

template <typename T>
struct first_non_void_type<T>
{
    using type_t = T;
};

template <typename... TsT>
struct first_non_void_type_error_check : first_non_void_type<TsT...>  // NOLINT
{
    static_assert(!std::is_same_v<typename first_non_void_type<TsT...>::type_t, void>,
                  "Error: All types in the list are void");
};

template <typename T, typename... TsT>
using first_non_void_type_t = typename first_non_void_type_error_check<T, TsT...>::type_t;

}  // namespace mrc
