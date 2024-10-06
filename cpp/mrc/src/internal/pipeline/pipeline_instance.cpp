/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/pipeline/pipeline_instance.hpp"

#include "internal/pipeline/pipeline_definition.hpp"
#include "internal/pipeline/pipeline_resources.hpp"
#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/segment/segment_definition.hpp"
#include "internal/segment/segment_instance.hpp"
#include "internal/service.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <exception>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::pipeline {

PipelineInstance::PipelineInstance(std::shared_ptr<const PipelineDefinition> definition,
                                   resources::Manager& resources) :
  PipelineResources(resources),
  Service("pipeline::PipelineInstance"),
  m_definition(std::move(definition))
{
    CHECK(m_definition);
    m_joinable_future = m_joinable_promise.get_future().share();
}

PipelineInstance::~PipelineInstance()
{
    Service::call_in_destructor();
}

void PipelineInstance::update()
{
    for (const auto& [name, manifold] : m_manifolds)
    {
        manifold->update_inputs();
        manifold->update_outputs();
        manifold->start();
    }
    for (const auto& [address, segment] : m_segments)
    {
        segment->service_start();
        segment->service_await_live();
    }
    mark_joinable();
}

void PipelineInstance::remove_segment(const SegmentAddress& address)
{
    auto search = m_segments.find(address);
    CHECK(search != m_segments.end());
    m_segments.erase(search);
}

void PipelineInstance::join_segment(const SegmentAddress& address)
{
    auto search = m_segments.find(address);
    CHECK(search != m_segments.end());
    search->second->service_await_join();
}

void PipelineInstance::stop_segment(const SegmentAddress& address)
{
    auto search = m_segments.find(address);
    CHECK(search != m_segments.end());

    auto [id, rank]    = segment_address_decode(address);
    const auto& segdef = m_definition->find_segment(id);

    for (const auto& name : segdef->ingress_port_names())
    {
        DVLOG(3) << "Dropping IngressPort for " << ::mrc::segment::info(address) << " on manifold " << name;
        // manifold(name).drop_output(address);
    }

    search->second->service_stop();
}

void PipelineInstance::create_segment(const SegmentAddress& address, std::uint32_t partition_id)
{
    // perform our allocations on the numa domain of the intended target
    // CHECK_LT(partition_id, m_resources->host_resources().size());
    CHECK_LT(partition_id, resources().partition_count());
    resources()
        .partition(partition_id)
        .runnable()
        .main()
        .enqueue([this, address, partition_id] {
            auto search = m_segments.find(address);
            CHECK(search == m_segments.end());

            auto [id, rank] = segment_address_decode(address);
            auto definition = m_definition->find_segment(id);
            auto segment    = std::make_unique<segment::SegmentInstance>(definition, rank, *this, partition_id);

            for (const auto& name : definition->egress_port_names())
            {
                VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for egress port " << name;
                std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
                if (!manifold)
                {
                    VLOG(10) << ::mrc::segment::info(address) << " creating manifold for egress port " << name;
                    manifold          = segment->create_manifold(name);
                    m_manifolds[name] = manifold;
                }
                segment->attach_manifold(manifold);
            }

            for (const auto& name : definition->ingress_port_names())
            {
                VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for ingress port " << name;
                std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
                if (!manifold)
                {
                    VLOG(10) << ::mrc::segment::info(address) << " creating manifold for ingress port " << name;
                    manifold          = segment->create_manifold(name);
                    m_manifolds[name] = manifold;
                }
                segment->attach_manifold(manifold);
            }

            m_segments[address] = std::move(segment);
        })
        .get();
}

manifold::Interface& PipelineInstance::manifold(const PortName& port_name)
{
    auto manifold = get_manifold(port_name);
    CHECK(manifold);
    return *manifold;
}

std::shared_ptr<manifold::Interface> PipelineInstance::get_manifold(const PortName& port_name)
{
    auto search = m_manifolds.find(port_name);
    if (search == m_manifolds.end())
    {
        return nullptr;
    }

    return m_manifolds.at(port_name);
}

void PipelineInstance::mark_joinable()
{
    if (!m_joinable)
    {
        m_joinable = true;
        m_joinable_promise.set_value();
    }
}

void PipelineInstance::do_service_start() {}

void PipelineInstance::do_service_await_live()
{
    m_joinable_future.get();
}

void PipelineInstance::do_service_stop()
{
    mark_joinable();

    for (auto& [id, segment] : m_segments)
    {
        stop_segment(id);
    }
}

void PipelineInstance::do_service_kill()
{
    mark_joinable();
    for (auto& [id, segment] : m_segments)
    {
        stop_segment(id);
        segment->service_kill();
    }
}

void PipelineInstance::do_service_await_join()
{
    std::exception_ptr first_exception = nullptr;
    m_joinable_future.get();
    for (const auto& [address, segment] : m_segments)
    {
        try
        {
            segment->service_await_join();
        } catch (...)
        {
            if (first_exception == nullptr)
            {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception)
    {
        LOG(ERROR) << "pipeline::PipelineInstance - an exception was caught while awaiting on segments - rethrowing";
        std::rethrow_exception(std::move(first_exception));
    }
}

}  // namespace mrc::pipeline
