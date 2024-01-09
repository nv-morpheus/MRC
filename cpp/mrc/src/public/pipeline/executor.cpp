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
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/pipeline/segment.hpp"
#include "mrc/pipeline/system.hpp"

#include <memory>
#include <utility>

namespace mrc {

pipeline::SegmentMapping::SegmentMapping(std::string segment_name) : m_segment_name(std::move(segment_name)) {}

bool pipeline::SegmentMapping::is_enabled() const
{
    return m_is_enabled;
}

void pipeline::SegmentMapping::set_enabled(bool is_enabled)
{
    m_is_enabled = is_enabled;
}

pipeline::PipelineMapping::PipelineMapping(std::shared_ptr<IPipeline> pipeline)
{
    for (const auto& segment : pipeline->segments())
    {
        m_segment_mappings.emplace(segment->name(), segment->name());
    }
}

pipeline::SegmentMapping& pipeline::PipelineMapping::get_segment(const std::string& segment_name)
{
    return m_segment_mappings.at(segment_name);
}

const pipeline::SegmentMapping& pipeline::PipelineMapping::get_segment(const std::string& segment_name) const
{
    return m_segment_mappings.at(segment_name);
}

Executor::Executor() : m_impl(make_executor_impl(std::make_shared<Options>())) {}

Executor::Executor(std::shared_ptr<Options> options) : m_impl(make_executor_impl(options)) {}

Executor::~Executor() = default;

pipeline::PipelineMapping& Executor::register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline)
{
    return m_impl->register_pipeline(std::move(pipeline));
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

std::unique_ptr<pipeline::IExecutor> make_executor_impl(std::shared_ptr<Options> options)
{
    // Convert options to a system object first
    auto system = mrc::make_system(std::move(options));

    auto full_system = system::SystemDefinition::unwrap(std::move(system));

    return std::make_unique<executor::ExecutorDefinition>(std::move(full_system));
}

std::unique_ptr<pipeline::IExecutor> make_executor_impl(std::unique_ptr<pipeline::ISystem> system)
{
    auto full_system = system::SystemDefinition::unwrap(std::move(system));

    return std::make_unique<executor::ExecutorDefinition>(std::move(full_system));
}

}  // namespace mrc
