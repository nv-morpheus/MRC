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

#include "mrc/core/thread.hpp"

#include "mrc/coroutines/thread_pool.hpp"

#include <concepts>
#include <iomanip>
#include <sstream>
#include <thread>

namespace mrc::this_thread {

namespace {

std::string to_hex(std::integral auto i)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(sizeof(i) * 2) << std::hex << i;
    return ss.str();
}

std::string init_thread_id()
{
    const auto* thread_pool = coroutines::ThreadPool::from_current_thread();
    if (thread_pool == nullptr)
    {
        std::stringstream ss;
        ss << "sys/" << to_hex(std::hash<std::thread::id>()(std::this_thread::get_id()));
        return ss.str();
    }

    std::stringstream ss;
    ss << thread_pool->description() << "/" << thread_pool->get_thread_id();
    return ss.str();
}

}  // namespace

const std::string& get_id()
{
    static thread_local std::string id = init_thread_id();
    return id;
}

}  // namespace mrc::this_thread
