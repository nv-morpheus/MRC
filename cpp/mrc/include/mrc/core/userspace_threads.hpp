/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <boost/fiber/all.hpp>

namespace mrc::userspace_threads {

// Suppress naming conventions in this file to allow matching std and boost libraries
// NOLINTBEGIN(readability-identifier-naming)

using mutex = boost::fibers::mutex;

using recursive_mutex = boost::fibers::recursive_mutex;

using cv = boost::fibers::condition_variable;

using cv_any = boost::fibers::condition_variable_any;

using launch = boost::fibers::launch;

template <typename T>
using promise = boost::fibers::promise<T>;

template <typename T>
using future = boost::fibers::future<T>;

template <typename T>
using shared_future = boost::fibers::shared_future<T>;

template <typename SignatureT>
using packaged_task = boost::fibers::packaged_task<SignatureT>;

template <class Function, class... Args>
static auto async(Function&& f, Args&&... args)
{
    return boost::fibers::async(f, std::forward<Args>(args)...);
}

template <typename Rep, typename Period>
static void sleep_for(std::chrono::duration<Rep, Period> const& timeout_duration)
{
    boost::this_fiber::sleep_for(timeout_duration);
}

template <typename Clock, typename Duration>
static void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time_point)
{
    boost::this_fiber::sleep_until(sleep_time_point);
}

// NOLINTEND(readability-identifier-naming)

}  // namespace mrc::userspace_threads
