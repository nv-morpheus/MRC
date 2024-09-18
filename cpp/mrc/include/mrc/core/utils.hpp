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
#include <functional>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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
    explicit Unwinder(std::function<void()> unwind_fn) :
      m_unwind_fn(std::move(unwind_fn)),
      m_ctor_exception_count(std::uncaught_exceptions())
    {}

    ~Unwinder() noexcept(false)
    {
        if (!!m_unwind_fn)
        {
            try
            {
                m_unwind_fn();
            } catch (...)
            {
                if (std::uncaught_exceptions() > m_ctor_exception_count)
                {
                    LOG(ERROR) << "Error occurred during unwinder function, but another exception is active.";
                    std::terminate();
                }

                LOG(ERROR) << "Error occurred during unwinder function. Rethrowing";
                throw;
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
    // Stores the number of active exceptions during creation. If the number of active exceptions during destruction is
    // greater, we do not throw and log error and terminate
    int m_ctor_exception_count;
    std::function<void()> m_unwind_fn;
};

#define MRC_UNWIND(unwinder_name, function) mrc::Unwinder unwinder_name(function);

#define MRC_UNWIND_AUTO(function) MRC_UNWIND(MRC_UNIQUE_VAR_NAME(__un_obj_), function)

template <typename T>
std::pair<std::set<T>, std::set<T>> set_compare(const std::set<T>& cur_set, const std::set<T>& new_set)
{
    std::set<T> remove;
    std::set<T> create;

    // set difference to determine which channels to remove
    std::set_difference(cur_set.begin(),
                        cur_set.end(),
                        new_set.begin(),
                        new_set.end(),
                        std::inserter(remove, remove.end()));

    // set difference to determine which channels to add
    std::set_difference(new_set.begin(),
                        new_set.end(),
                        cur_set.begin(),
                        cur_set.end(),
                        std::inserter(create, create.end()));

    return std::make_pair(std::move(create), std::move(remove));
}
}  // namespace mrc
