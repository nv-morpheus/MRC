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

#pragma once

#include "mrc/core/addresses.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/manifold/connectable.hpp"
#include "mrc/manifold/factory.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/types.hpp"

#include <condition_variable>
#include <memory>
#include <string>
#include <utility>

namespace mrc::pipeline {
class ManifoldInstance;
}

namespace mrc::segment {

class SegmentInstance;

class EgressPortBase : public runnable::Launchable, public manifold::Connectable, public virtual ObjectProperties
{
  private:
    virtual edge::IWritableAcceptorBase& get_downstream_source() const = 0;

    friend class mrc::pipeline::ManifoldInstance;
};

template <typename T>
class EgressPort final : public Object<node::RxSinkBase<T>>,
                         public EgressPortBase,
                         public std::enable_shared_from_this<EgressPort<T>>
{
    // debug tap
    // rxcpp::operators::tap([this](const T& t) {
    //     LOG(INFO) << segment::info(m_segment_address) << " egress port " << m_port_name << ": tap = " << t;
    // })

  public:
    EgressPort(SegmentAddress2 segment_address, PortName name) :
      m_port_address(PortAddress2(segment_address.executor_id,
                                  segment_address.pipeline_id,
                                  segment_address.segment_id,
                                  port_name_hash(name))),
      m_port_name(std::move(name)),
      m_node(std::make_unique<node::RxNode<T>>(rxcpp::operators::map([this](T data) {
                                                   return data;
                                               }),
                                               rxcpp::operators::finally([this]() {
                                                   VLOG(10) << "EgressPort " << m_port_name
                                                            << " completed. Dropping connection";
                                               })))
    {}

  private:
    node::RxSinkBase<T>* get_object() const final
    {
        CHECK(m_node) << "failed to acquire backing runnable for egress port " << m_port_name;
        return m_node.get();
    }

    std::unique_ptr<runnable::Launcher> prepare_launcher(runnable::LaunchControl& launch_control) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        CHECK(m_node);
        // CHECK(m_manifold_connected) << "manifold not set for egress port";
        return launch_control.prepare_launcher(std::move(m_node));
    }

    std::shared_ptr<manifold::Interface> make_manifold(runnable::IRunnableResources& resources) final
    {
        return manifold::Factory<T>::make_manifold(m_port_name, resources);
    }

    void connect_to_manifold(std::shared_ptr<manifold::Interface> manifold) final
    {
        // egress ports connect to manifold inputs
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        DCHECK_EQ(manifold->port_name(), m_port_name);
        CHECK(m_node);
        CHECK(!m_manifold_connected);
        manifold->add_local_input(m_port_address, m_node.get());
        m_manifold_connected = true;
    }

    edge::IWritableAcceptorBase& get_downstream_source() const override
    {
        return *m_node;
    }

    PortAddress2 m_port_address;
    PortName m_port_name;
    std::unique_ptr<node::RxNode<T>> m_node;
    bool m_manifold_connected{false};
    runnable::LaunchOptions m_launch_options;
    std::mutex m_mutex;
};

}  // namespace mrc::segment
