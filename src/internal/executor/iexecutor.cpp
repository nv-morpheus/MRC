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

#include "internal/executor/executor.hpp"
#include "internal/system/system.hpp"

#include <srf/internal/executor/iexecutor.hpp>
#include <srf/internal/system/isystem.hpp>
#include <srf/options/options.hpp>

#include <glog/logging.h>

#include <utility>

namespace srf::internal::executor {

IExecutor::IExecutor(std::shared_ptr<Options> options) : m_impl(make_executor(std::move(options))) {}
IExecutor::IExecutor(std::unique_ptr<system::ISystem> system) : m_impl(make_executor(system::System::unwrap(*system)))
{
    CHECK(m_impl);
}

IExecutor::~IExecutor() = default;

void IExecutor::register_pipeline(std::unique_ptr<internal::pipeline::IPipeline> pipeline)
{
    CHECK(m_impl);
    m_impl->register_pipeline(std::move(pipeline));
}

void IExecutor::start()
{
    CHECK(m_impl);
    m_impl->service_start();
}

void IExecutor::stop()
{
    CHECK(m_impl);
    m_impl->service_stop();
}

void IExecutor::join()
{
    CHECK(m_impl);
    m_impl->service_await_join();
}

}  // namespace srf::internal::executor
