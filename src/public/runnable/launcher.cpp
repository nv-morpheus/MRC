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

#include "srf/runnable/launcher.hpp"

#include "srf/runnable/context.hpp"
#include "srf/runnable/engine.hpp"
#include "srf/runnable/runner.hpp"

#include <glog/logging.h>

#include <ostream>
#include <utility>

namespace srf::runnable {

Launcher::Launcher(std::unique_ptr<Runner> runner,
                   std::vector<std::shared_ptr<Context>>&& contexts,
                   std::shared_ptr<Engines> engines) :
  m_runner(std::move(runner)),
  m_contexts(std::move(contexts)),
  m_engines(std::move(engines))
{}

Launcher::~Launcher() = default;

std::unique_ptr<Runner> Launcher::ignition()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    CHECK(m_runner);
    CHECK(m_engines);
    CHECK(m_contexts.size());
    m_runner->enqueue(m_engines, std::move(m_contexts));
    return std::move(m_runner);
}

void Launcher::apply(std::function<void(Runner&)> fn)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    CHECK(m_runner);
    fn(*m_runner);
}

}  // namespace srf::runnable
