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

#include "internal/segment/instance.hpp"

#include "internal/resources/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/segment/builder.hpp"
#include "internal/segment/definition.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/segment/egress_port.hpp"
#include "mrc/segment/ingress_port.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>

namespace mrc::internal::segment {

Instance::Instance(std::shared_ptr<const Definition> definition,
                   SegmentRank rank,
                   pipeline::Resources& resources,
                   std::size_t partition_id) :
  m_name(definition->name()),
  m_id(definition->id()),
  m_rank(rank),
  m_address(segment_address_encode(m_id, m_rank)),
  m_info(::mrc::segment::info(segment_address_encode(m_id, rank))),
  m_resources(resources),
  m_default_partition_id(partition_id)
{
    // construct the segment definition on the intended numa node
    m_builder = m_resources.resources()
                    .partition(m_default_partition_id)
                    .runnable()
                    .main()
                    .enqueue([&]() mutable {
                        return std::make_unique<Builder>(definition, rank, m_resources, m_default_partition_id);
                    })
                    .get();
}

const std::string& Instance::name() const
{
    return m_name;
}

const SegmentID& Instance::id() const
{
    return m_id;
}

const SegmentRank& Instance::rank() const
{
    return m_rank;
}

const SegmentAddress& Instance::address() const
{
    return m_address;
}

void Instance::do_service_start()
{
    // prepare launchers from m_builder
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> m_launchers;
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> m_egress_launchers;
    std::map<std::string, std::unique_ptr<mrc::runnable::Launcher>> m_ingress_launchers;

    auto apply_callback = [this](std::unique_ptr<mrc::runnable::Launcher>& launcher, std::string name) {
        launcher->apply([this, n = std::move(name)](mrc::runnable::Runner& runner) {
            runner.on_completion_callback([this, n](bool ok) {
                if (!ok)
                {
                    DVLOG(10) << info() << ": detected a failure in node " << n << "; issuing service_kill()";
                    service_kill();
                }
            });
        });
    };

    for (const auto& [name, node] : m_builder->nodes())
    {
        DVLOG(10) << info() << " constructing launcher for " << name;
        m_launchers[name] = node->prepare_launcher(
            m_resources.resources().partition(m_default_partition_id).runnable().launch_control());
        apply_callback(m_launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->egress_ports())
    {
        DVLOG(10) << info() << " constructing launcher egress port " << name;
        m_egress_launchers[name] = node->prepare_launcher(
            m_resources.resources().partition(m_default_partition_id).runnable().launch_control());
        apply_callback(m_egress_launchers[name], name);
    }

    for (const auto& [name, node] : m_builder->ingress_ports())
    {
        DVLOG(10) << info() << " constructing launcher ingress port " << name;
        m_ingress_launchers[name] = node->prepare_launcher(
            m_resources.resources().partition(m_default_partition_id).runnable().launch_control());
        apply_callback(m_ingress_launchers[name], name);
    }

    DVLOG(10) << info() << " issuing start request";

    for (const auto& [name, launcher] : m_egress_launchers)
    {
        DVLOG(10) << info() << " launching egress port " << name;
        m_egress_runners[name] = launcher->ignition();
    }

    for (const auto& [name, launcher] : m_launchers)
    {
        DVLOG(10) << info() << " launching node " << name;
        m_runners[name] = launcher->ignition();
    }

    for (const auto& [name, launcher] : m_ingress_launchers)
    {
        DVLOG(10) << info() << " launching ingress port " << name;
        m_ingress_runners[name] = launcher->ignition();
    }

    m_egress_launchers.clear();
    m_launchers.clear();
    m_ingress_launchers.clear();

    DVLOG(10) << info() << " start has been initiated; use the is_running future to await on startup";
}

void Instance::do_service_stop()
{
    DVLOG(10) << info() << " issuing stop request";

    // we do not issue stop for port since they are nodes and stop has no effect

    for (const auto& [name, runner] : m_runners)
    {
        DVLOG(10) << info() << " issuing stop for node " << name;
        runner->stop();
    }

    DVLOG(10) << info() << " stop has been initiated; use the is_completed future to await on shutdown";
}

void Instance::do_service_kill()
{
    DVLOG(10) << info() << " issuing kill request";

    for (const auto& [name, runner] : m_ingress_runners)
    {
        DVLOG(10) << info() << " issuing kill for ingress port " << name;
        runner->kill();
    }

    for (const auto& [name, runner] : m_runners)
    {
        DVLOG(10) << info() << " issuing kill for node " << name;
        runner->kill();
    }

    for (const auto& [name, runner] : m_egress_runners)
    {
        DVLOG(10) << info() << " issuing kill for egress port " << name;
        runner->kill();
    }

    DVLOG(10) << info() << " kill has been initiated; use the is_completed future to await on shutdown";
}

void Instance::do_service_await_live()
{
    DVLOG(10) << info() << " await_live started";
    for (const auto& [name, runner] : m_ingress_runners)
    {
        DVLOG(10) << info() << " awaiting on ingress port " << name;
        runner->await_live();
    }
    for (const auto& [name, runner] : m_runners)
    {
        DVLOG(10) << info() << " awaiting on  " << name;
        runner->await_live();
    }
    for (const auto& [name, runner] : m_egress_runners)
    {
        DVLOG(10) << info() << " awaiting on egress port " << name;
        runner->await_live();
    }
    DVLOG(10) << info() << " join complete";
}

void Instance::do_service_await_join()
{
    DVLOG(10) << info() << " join started";
    std::exception_ptr first_exception = nullptr;

    auto check = [&first_exception](mrc::runnable::Runner& runner) {
        try
        {
            runner.await_join();
        } catch (...)
        {
            if (first_exception == nullptr)
            {
                first_exception = std::current_exception();
            }
        }
    };

    for (const auto& [name, runner] : m_ingress_runners)
    {
        DVLOG(10) << info() << " awaiting on ingress port join to " << name;
        check(*runner);
    }
    for (const auto& [name, runner] : m_runners)
    {
        DVLOG(10) << info() << " awaiting on join to " << name;
        check(*runner);
    }
    for (const auto& [name, runner] : m_egress_runners)
    {
        DVLOG(10) << info() << " awaiting on egress port join to " << name;
        check(*runner);
    }
    DVLOG(10) << info() << " join complete";
    if (first_exception)
    {
        LOG(ERROR) << "segment::Instance - an exception was caught while awaiting on one or more nodes - rethrowing";
        rethrow_exception(std::move(first_exception));
    }
}

void Instance::attach_manifold(std::shared_ptr<manifold::Interface> manifold)
{
    auto port_name = manifold->port_name();

    {
        auto search = m_builder->egress_ports().find(port_name);
        if (search != m_builder->egress_ports().end())
        {
            DVLOG(10) << info() << " attaching manifold for egress port " << port_name;
            search->second->connect_to_manifold(std::move(manifold));
            return;
        }
    }

    {
        auto search = m_builder->ingress_ports().find(port_name);
        if (search != m_builder->ingress_ports().end())
        {
            DVLOG(10) << info() << " attaching manifold for ingress port " << port_name;
            search->second->connect_to_manifold(std::move(manifold));
            return;
        }
    }

    LOG(ERROR) << "unable to find an ingress or egress port matching the port_name " << port_name;
    throw exceptions::MrcRuntimeError("invalid manifold for segment");
}

const std::string& Instance::info() const
{
    return m_info;
}

std::shared_ptr<manifold::Interface> Instance::create_manifold(const PortName& name)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    DVLOG(10) << info() << " attempting to build manifold for port " << name;
    {
        auto search = m_builder->egress_ports().find(name);
        if (search != m_builder->egress_ports().end())
        {
            return search->second->make_manifold(m_resources.resources().partition(m_default_partition_id).runnable());
        }
    }
    {
        auto search = m_builder->ingress_ports().find(name);
        if (search != m_builder->ingress_ports().end())
        {
            return search->second->make_manifold(m_resources.resources().partition(m_default_partition_id).runnable());
        }
    }
    LOG(FATAL) << info() << " unable to match ingress or egress port name";
    return nullptr;
}

}  // namespace mrc::internal::segment
