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

#include "internal/runnable/engines.hpp"

#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/launch_options.hpp"

#include <utility>

namespace mrc::internal::runnable {

Engines::Engines(LaunchOptions launch_options) : m_launch_options(std::move(launch_options)) {}

const std::vector<std::shared_ptr<::mrc::runnable::Engine>>& Engines::launchers() const
{
    return m_launchers;
}

std::size_t Engines::size() const
{
    return m_launchers.size();
}

const LaunchOptions& Engines::launch_options() const
{
    return m_launch_options;
}

void Engines::add_launcher(std::shared_ptr<::mrc::runnable::Engine> launcher)
{
    m_launchers.push_back(std::move(launcher));
}

void Engines::clear_launchers()
{
    m_launchers.clear();
}

}  // namespace mrc::internal::runnable
