/**
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace mrc {

struct userspace_threads  // NOLINT
{
    using mutex = boost::fibers::mutex;  // NOLINT

    using cv = boost::fibers::condition_variable;  // NOLINT

    using launch = boost::fibers::launch;  // NOLINT

    template <typename T>
    using promise = boost::fibers::promise<T>;  // NOLINT

    template <typename T>
    using future = boost::fibers::future<T>;  // NOLINT

    template <typename T>
    using shared_future = boost::fibers::shared_future<T>;  // NOLINT

    template <class R, class... Args>                                // NOLINT
    using packaged_task = boost::fibers::packaged_task<R(Args...)>;  // NOLINT

    template <class Function, class... Args>  // NOLINT
    static auto async(Function&& f, Args&&... args)
    {
        return boost::fibers::async(f, std::forward<Args>(args)...);
    }

    template <typename Rep, typename Period>  // NOLINT
    static void sleep_for(std::chrono::duration<Rep, Period> const& timeout_duration)
    {
        boost::this_fiber::sleep_for(timeout_duration);
    }

    template <typename Clock, typename Duration>  // NOLINT
    static void sleep_until(std::chrono::time_point<Clock, Duration> const& sleep_time_point)
    {
        boost::this_fiber::sleep_until(sleep_time_point);
    }
};
}  // namespace mrc
