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

#include "mrc/constants.hpp"
#include "mrc/core/bitmap.hpp"
#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/fiber_pool.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// work-around for known iwyu issue
// https://github.com/include-what-you-use/include-what-you-use/issues/908
// IWYU pragma: no_include <algorithm>

namespace mrc::runnable {

class Runner;

/**
 * @brief Provides a one-time use method to execute a task on a supplied execution context
 */
class Engine
{
  public:
    virtual ~Engine() = default;

    virtual EngineType engine_type() const = 0;

  private:
    virtual Future<void> launch_task(std::function<void()> task) = 0;

    friend Runner;
};

/**
 * @brief Provides a set of Engines that
 *
 */
class Engines
{
  public:
    virtual ~Engines() = default;

    virtual const std::vector<std::shared_ptr<Engine>>& launchers() const = 0;
    virtual const LaunchOptions& launch_options() const                   = 0;
    virtual EngineType engine_type() const                                = 0;
    virtual std::size_t size() const                                      = 0;
};

}  // namespace mrc::runnable
