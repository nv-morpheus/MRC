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

#include "srf/utils/macros.hpp"

#include <algorithm>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <unordered_set>

namespace srf {

std::uint16_t hash(const std::string&);

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

    Unwinder()                           = delete;
    Unwinder(const Unwinder&)            = delete;
    Unwinder& operator=(const Unwinder&) = delete;

  private:
    FunctionT* m_function;
};

#define SRF_UNWIND(var_name, function) SRF_UNWIND_EXPLICIT(uw_func_##var_name, var_name, function)

#define SRF_UNWIND_AUTO(function) \
    SRF_UNWIND_EXPLICIT(SRF_UNIQUE_VAR_NAME(uw_func_), SRF_UNIQUE_VAR_NAME(un_obj_), function)

#define SRF_UNWIND_EXPLICIT(function_name, unwinder_name, function) \
    auto function_name = (function);                                \
    srf::Unwinder<decltype(function_name)> unwinder_name(std::addressof(function_name))

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
}  // namespace srf
