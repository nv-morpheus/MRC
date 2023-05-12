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

#include "mrc/engine/system/iresources.hpp"
#include "mrc/options/options.hpp"
#include "mrc/pipeline/pipeline.hpp"

#include <memory>

namespace mrc {

Executor::Executor() : Executor(std::make_shared<Options>()) {}

Executor::Executor(std::shared_ptr<Options> options) : m_impl(make_executor(options)) {}

Executor::Executor(std::unique_ptr<system::IResources> resources) : m_impl(make_executor(std::move(resources))) {}

Executor::~Executor() = default;

void Executor::register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline)
{
    m_impl->register_pipeline(std::move(pipeline));
}

void Executor::start()
{
    m_impl->start();
}

void Executor::stop()
{
    m_impl->stop();
}

void Executor::join()
{
    m_impl->join();
}

std::unique_ptr<pipeline::IExecutor> make_executor(std::shared_ptr<Options> options)
{
    return std::make_unique<Executor>(std::move(options));
}

std::unique_ptr<pipeline::IExecutor> make_executor(std::unique_ptr<system::IResources> resources)
{
    return std::make_unique<Executor>(std::move(resources));
}

}  // namespace mrc
