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
#include "internal/pipeline/port_graph.hpp"
#include "internal/pipeline/types.hpp"
#include "internal/resources/manager.hpp"
#include "internal/system/system.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/options/options.hpp"

#include <glog/logging.h>

#include <map>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>

namespace mrc::internal::executor {

static bool valid_pipeline(const pipeline::Pipeline& pipeline);

Executor::Executor(std::shared_ptr<Options> options) :
  SystemProvider(system::make_system(std::move(options))),
  m_resources_manager(std::make_unique<resources::Manager>(*this))
{}

Executor::Executor(std::unique_ptr<system::Resources> resources) :
  SystemProvider(*resources),
  m_resources_manager(std::make_unique<resources::Manager>(std::move(resources)))
{}

Executor::~Executor()
{
    Service::call_in_destructor();
}

void Executor::register_pipeline(std::unique_ptr<pipeline::IPipeline> ipipeline)
{
    CHECK(ipipeline);
    CHECK(m_pipeline_manager == nullptr);

    auto pipeline = pipeline::Pipeline::unwrap(*ipipeline);

    if (!valid_pipeline(*pipeline))
    {
        throw exceptions::MrcRuntimeError("pipeline validation failed");
    }

    m_pipeline_manager = std::make_unique<pipeline::Manager>(pipeline, *m_resources_manager);
}

void Executor::do_service_start()
{
    CHECK(m_pipeline_manager);
    m_pipeline_manager->service_start();

    pipeline::SegmentAddresses initial_segments;
    for (const auto& [id, segment] : m_pipeline_manager->pipeline().segments())
    {
        auto address              = segment_address_encode(id, 0);  // rank 0
        initial_segments[address] = 0;                              // partition 0;
    }
    m_pipeline_manager->push_updates(std::move(initial_segments));
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

// convert to std::expect
bool valid_pipeline(const pipeline::Pipeline& pipeline)
{
    bool valid = true;
    pipeline::PortGraph pg(pipeline);

    for (const auto& [name, connections] : pg.port_map())
    {
        // first validate all port names have at least one:
        // - segment using that port as an ingress, and
        // - segment using that port as an egress
        if (connections.egress_segments.empty() or connections.ingress_segments.empty())
        {
            valid = false;
            // todo - print list of segments names for ingress/egres connections to this port
            LOG(WARNING) << "port: " << name << " has incomplete connections - used as ingress on "
                         << connections.ingress_segments.size() << " segments; used as egress on "
                         << connections.egress_segments.size() << " segments";
        }

        // we currently only have an load-balancer manifold
        // it doesn't make sense to connect segments of different types to a load-balancer, they should probably be
        // broadcast
        // in general, if there are more than one type of segments writing to or reading from a manifold, then that port
        // should have an explicit manifold type specified
        if (connections.egress_segments.size() > 1 or connections.ingress_segments.size() > 1)
        {
            valid = false;
            LOG(WARNING) << "port: " << name
                         << " has more than 1 segment type connected to an ingress or egress port; this is currently "
                            "an invalid configuration as there are no manifold available to handle this condition";
        }
    }

    return valid;
}

}  // namespace mrc::internal::executor
