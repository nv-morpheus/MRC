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

#pragma once

#include "mrc/manifold/connectable.hpp"
#include "mrc/manifold/factory.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/generic_node.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/sink_channel.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/segment/object.hpp"

#include <condition_variable>
#include <memory>
#include <string>

namespace mrc::segment {

class Instance;

struct IngressPortBase : public runnable::Launchable, public manifold::Connectable, public virtual ObjectProperties
{
    friend Instance;
};

template <typename T>
class IngressPort : public Object<node::SourceProperties<T>>, public IngressPortBase
{
    // tap for debugging
    // rxcpp::operators::tap([this](const T& t) {
    //     LOG(INFO) << segment::info(m_segment_address) << "ingress port " << m_port_name << ": tap = " << t;
    // })

  public:
    IngressPort(SegmentAddress address, PortName name) :
      m_segment_address(address),
      m_port_name(std::move(name)),
      m_source(std::make_unique<node::RxNode<T>>())
    {
        this->set_name(m_port_name);
    }

  private:
    node::SourceProperties<T>* get_object() const final
    {
        CHECK(m_source);
        return m_source.get();
    }

    std::unique_ptr<runnable::Launcher> prepare_launcher(runnable::LaunchControl& launch_control) final
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        CHECK(m_source);
        return launch_control.prepare_launcher(std::move(m_source));
    }

    std::shared_ptr<manifold::Interface> make_manifold(pipeline::Resources& resources) final
    {
        return manifold::Factory<T>::make_manifold(m_port_name, resources);
    }

    void connect_to_manifold(std::shared_ptr<manifold::Interface> manifold) final
    {
        // ingress ports connect to manifold outputs
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        CHECK(m_source);
        manifold->add_output(m_segment_address, m_source.get());
    }

    SegmentAddress m_segment_address;
    PortName m_port_name;
    std::unique_ptr<node::RxNode<T>> m_source;
    std::mutex m_mutex;

    friend Instance;
};

}  // namespace mrc::segment
