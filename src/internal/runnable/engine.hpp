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

#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"

#include <functional>
#include <mutex>

namespace mrc::internal::runnable {

using ::mrc::runnable::EngineType;

class Engine : public ::mrc::runnable::Engine
{
    Future<void> launch_task(std::function<void()> task) final;

    virtual Future<void> do_launch_task(std::function<void()> task) = 0;

    bool m_launched{false};
    std::mutex m_mutex;
};

}  // namespace mrc::internal::runnable
