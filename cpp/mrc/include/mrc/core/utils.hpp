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

#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mrc {

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
class Unwinder
{
  public:
    explicit Unwinder(std::function<void()> unwind_fn) : m_unwind_fn(std::move(unwind_fn)) {}

    ~Unwinder()
    {
        if (!!m_unwind_fn)
        {
            try
            {
                m_unwind_fn();
            } catch (...)
            {
                LOG(ERROR) << "Fatal error during unwinder function";
                std::terminate();
            }
        }
    }

    void detach()
    {
        m_unwind_fn = nullptr;
    }

    Unwinder()                           = delete;
    Unwinder(const Unwinder&)            = delete;
    Unwinder& operator=(const Unwinder&) = delete;

    static Unwinder create(std::function<void()> unwind_fn)
    {
        return Unwinder(std::move(unwind_fn));
    }

  private:
    std::function<void()> m_unwind_fn;
};

#define MRC_UNWIND(unwinder_name, function) mrc::Unwinder unwinder_name(function);

#define MRC_UNWIND_AUTO(function) MRC_UNWIND(MRC_UNIQUE_VAR_NAME(__un_obj_), function)

}  // namespace mrc
