/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/task.hpp"
#include "mrc/coroutines/time.hpp"

#include <coroutine>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::coroutines {

/**
 * @brief Scheduler base class
 */
class Scheduler : public std::enable_shared_from_this<Scheduler>
{
  public:
    virtual ~Scheduler() = default;

    /**
     * @brief Resumes a coroutine according to the scheduler's implementation.
     */
    virtual void resume(std::coroutine_handle<> handle) noexcept = 0;

    /**
     * @brief Suspends the current function and resumes it according to the scheduler's implementation.
     */
    [[nodiscard]] virtual Task<> yield() = 0;

    /**
     * @brief Suspends the current function for a given duration and resumes it according to the schedulers's
     * implementation.
     */
    [[nodiscard]] virtual Task<> yield_for(std::chrono::milliseconds amount) = 0;

    /**
     * @brief Suspends the current function until a given time point and resumes it according to the schedulers's
     * implementation.
     */
    [[nodiscard]] virtual Task<> yield_until(time_point_t time) = 0;
};

}  // namespace mrc::coroutines
