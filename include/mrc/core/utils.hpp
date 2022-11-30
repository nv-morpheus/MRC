/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/utils/macros.hpp"

#include <algorithm>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

namespace mrc {

std::uint16_t hash(const std::string&);

template <typename T>
constexpr auto type_name() noexcept
{
    std::string_view name = "[with T = <UnsupportedType>]";
    std::string_view prefix;
    std::string_view suffix;
#ifdef __clang__
    name       = __PRETTY_FUNCTION__;
    auto start = name.find_first_of('[');
    auto end   = name.find_last_of(']');

    name = name.substr(start, end - start + 1);
#elif defined(__GNUC__)
    name       = __PRETTY_FUNCTION__;
    auto start = name.find_first_of('[');
    auto end   = name.find_last_of(']');

    name = name.substr(start, end - start + 1);
#elif defined(_MSC_VER)
    name   = __FUNCSIG__;
    prefix = "auto __cdecl type_name<";
    suffix = ">(void) noexcept";

    name.remove_prefix(prefix.size());
    name.remove_suffix(suffix.size());
#endif

    return name;
}

template <typename KeyT, typename ValT>
std::set<KeyT> extract_keys(const std::unordered_map<KeyT, ValT>& stdmap)
{
    std::set<KeyT> keys;
    for (const auto& pair : stdmap)
    {
        keys.insert(pair.first);
    }
    return keys;
}

template <typename KeyT, typename ValT>
std::set<KeyT> extract_keys(const std::map<KeyT, ValT>& stdmap)
{
    std::set<KeyT> keys;
    for (const auto& pair : stdmap)
    {
        keys.insert(pair.first);
    }
    return keys;
}

// RAII will execute a function when destroyed.
template <typename FunctionT>
class Unwinder
{
  public:
    ~Unwinder()
    {
        if (!!m_function)
        {
            try
            {
                (*m_function)();
            } catch (...)
            {
                std::terminate();
            }
        }
    }

    explicit Unwinder(FunctionT* function_arg) : m_function(function_arg) {}

    void detach()
    {
        m_function = nullptr;
    }

    Unwinder()                = delete;
    Unwinder(const Unwinder&) = delete;
    Unwinder& operator=(const Unwinder&) = delete;

  private:
    FunctionT* m_function;
};

#define MRC_UNWIND(var_name, function) MRC_UNWIND_EXPLICIT(uw_func_##var_name, var_name, function)

#define MRC_UNWIND_AUTO(function) \
    MRC_UNWIND_EXPLICIT(MRC_UNIQUE_VAR_NAME(uw_func_), MRC_UNIQUE_VAR_NAME(un_obj_), function)

#define MRC_UNWIND_EXPLICIT(function_name, unwinder_name, function) \
    auto function_name = (function);                                \
    mrc::Unwinder<decltype(function_name)> unwinder_name(std::addressof(function_name))

template <typename T>
std::pair<std::set<T>, std::set<T>> set_compare(const std::set<T>& cur_set, const std::set<T>& new_set)
{
    std::set<T> remove;
    std::set<T> create;

    // set difference to determine which channels to remove
    std::set_difference(
        cur_set.begin(), cur_set.end(), new_set.begin(), new_set.end(), std::inserter(remove, remove.end()));

    // set difference to determine which channels to add
    std::set_difference(
        new_set.begin(), new_set.end(), cur_set.begin(), cur_set.end(), std::inserter(create, create.end()));

    return std::make_pair(std::move(create), std::move(remove));
}
}  // namespace mrc
