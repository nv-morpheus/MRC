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

#include "internal/runnable/thread_engine.hpp"

#include "internal/system/system.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/packaged_task.hpp>

#include <type_traits>
#include <utility>

namespace srf::internal::runnable {

ThreadEngine::ThreadEngine(CpuSet cpu_set, std::shared_ptr<system::System> system) :
  m_cpu_set(std::move(cpu_set)),
  m_system(std::move(system))
{}

ThreadEngine::~ThreadEngine()
{
    m_thread->join();
}

std::thread::id ThreadEngine::get_id() const
{
    return m_thread->get_id();
}

Future<void> ThreadEngine::do_launch_task(std::function<void()> task)
{
    boost::fibers::packaged_task<void()> pkg_task(std::move(task));
    auto future = pkg_task.get_future();
    m_thread    = std::make_unique<std::thread>(m_system->make_thread("thread_engine", m_cpu_set, std::move(pkg_task)));
    return std::move(future);
}

runnable::EngineType ThreadEngine::engine_type() const
{
    return EngineType::Thread;
}

}  // namespace srf::internal::runnable
