/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runnable/engine.hpp"
#include "internal/system/threading_resources.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/types.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <thread>

namespace mrc::system {
class Thread;
}  // namespace mrc::system
namespace mrc::runnable {
enum class EngineType;
}  // namespace mrc::runnable

namespace mrc::runnable {

class ThreadEngine final : public Engine
{
  public:
    explicit ThreadEngine(CpuSet cpu_set, const system::ThreadingResources& system);
    ~ThreadEngine() final;

    EngineType engine_type() const final;

  protected:
    std::optional<std::thread::id> get_id() const;

  private:
    Future<void> do_launch_task(std::function<void()> task) final;

    CpuSet m_cpu_set;
    const system::ThreadingResources& m_system;
    std::unique_ptr<system::Thread> m_thread;
};

}  // namespace mrc::runnable
