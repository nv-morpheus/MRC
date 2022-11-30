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

#include "mrc/runnable/context.hpp"
#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/utils/macros.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace mrc::runnable {

/**
 * @brief This is one-time use object used to launch a Runnable.
 */
class Launcher final
{
  public:
    Launcher(std::unique_ptr<Runner> runner,
             std::vector<std::shared_ptr<Context>>&& contexts,
             std::shared_ptr<Engines> engines);

    ~Launcher();

    DELETE_COPYABILITY(Launcher);
    DELETE_MOVEABILITY(Launcher);

    /**
     * @brief Launches the Runnable
     * @note Launcher, nor LaunchControl take ownership of the launched Runnable. This is the caller's responsibility.
     * @return std::unique_ptr<Runner>
     */
    std::unique_ptr<Runner> ignition();

    /**
     * @brief Access the Runners's mutable API while holding the Launchers lock
     *
     * @param fn
     */
    void apply(std::function<void(Runner&)> fn);

  private:
    std::unique_ptr<Runner> m_runner;
    std::vector<std::shared_ptr<Context>> m_contexts;
    std::shared_ptr<Engines> m_engines;
    std::mutex m_mutex;
};

}  // namespace mrc::runnable
