/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/pipeline/executor.hpp"

#include "internal/executor/executor_definition.hpp"
#include "internal/system/system.hpp"

#include "mrc/options/options.hpp"
#include "mrc/pipeline/system.hpp"

#include <glog/logging.h>

#include <memory>
#include <utility>

namespace mrc {

Executor::Executor() : m_impl(make_executor(std::make_shared<Options>())) {}

Executor::Executor(std::shared_ptr<Options> options, std::function<void(State)> state_change_cb) :
  m_impl(make_executor(options)),
  m_state_change_cb(std::move(state_change_cb))
{}

Executor::~Executor() = default;

void Executor::register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline)
{
    m_impl->register_pipeline(std::move(pipeline));
}

void Executor::change_stage(State new_state)
{
    DVLOG(1) << "mrc::Executor - Changing state to " << static_cast<int>(new_state);
    m_state = new_state;
    if (m_state_change_cb)
    {
        m_state_change_cb(m_state);
    }
}

void Executor::start()
{
    m_impl->start();
    change_stage(State::Run);
}

void Executor::stop()
{
    m_impl->stop();
    change_stage(State::Stop);
}

void Executor::join()
{
    m_impl->join();
    change_stage(State::Joined);
}

std::unique_ptr<pipeline::IExecutor> make_executor(std::shared_ptr<Options> options,
                                                   std::function<void(State)> state_change_cb)
{
    // Convert options to a system object first
    auto system = mrc::make_system(std::move(options));

    auto full_system = system::SystemDefinition::unwrap(std::move(system));

    return std::make_unique<executor::ExecutorDefinition>(std::move(full_system));
}

std::unique_ptr<pipeline::IExecutor> make_executor(std::unique_ptr<pipeline::ISystem> system,
                                                   std::function<void(State)> state_change_cb)
{
    auto full_system = system::SystemDefinition::unwrap(std::move(system));

    return std::make_unique<executor::ExecutorDefinition>(std::move(full_system));
}

}  // namespace mrc
