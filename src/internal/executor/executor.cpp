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

#include "internal/pipeline/manager.hpp"
#include "internal/pipeline/pipeline.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/system/system.hpp"
#include "srf/core/addresses.hpp"
#include "srf/internal/pipeline/ipipeline.hpp"
#include "srf/options/options.hpp"
#include "srf/types.hpp"

#include <glog/logging.h>

#include <map>

namespace srf::internal::executor {

Executor::Executor(Handle<Options> options) :
  SystemProvider(system::make_system(std::move(options))),
  m_resources_manager(std::make_unique<resources::Manager>(*this))
{}

Executor::Executor(Handle<system::System> system) :
  SystemProvider(std::move(system)),
  m_resources_manager(std::make_unique<resources::Manager>(*this))
{}

Executor::~Executor()
{
    Service::call_in_destructor();
}

void Executor::register_pipeline(std::unique_ptr<pipeline::IPipeline> ipipeline)
{
    CHECK(ipipeline);
    CHECK(m_pipeline_manager == nullptr);

    auto pipeline      = pipeline::Pipeline::unwrap(*ipipeline);
    m_pipeline_manager = std::make_unique<pipeline::Manager>(pipeline, *m_resources_manager);

    pipeline::SegmentAddresses initial_segments;
    for (const auto& [id, segment] : pipeline->segments())
    {
        auto address              = segment_address_encode(id, 0);  // rank 0
        initial_segments[address] = 0;                              // partition 0;
    }
    m_pipeline_manager->push_updates(std::move(initial_segments));
}

void Executor::do_service_start()
{
    CHECK(m_pipeline_manager);
    m_pipeline_manager->service_start();
}
void Executor::do_service_stop()
{
    CHECK(m_pipeline_manager);
    m_pipeline_manager->service_stop();
}
void Executor::do_service_kill()
{
    CHECK(m_pipeline_manager);
    return m_pipeline_manager->service_kill();
}
void Executor::do_service_await_live()
{
    CHECK(m_pipeline_manager);
    m_pipeline_manager->service_await_live();
}
void Executor::do_service_await_join()
{
    CHECK(m_pipeline_manager);
    m_pipeline_manager->service_await_join();
}

}  // namespace srf::internal::executor
