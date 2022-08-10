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

#include "srf/core/addresses.hpp"
#include "srf/manifold/composite_manifold.hpp"
#include "srf/manifold/interface.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/generic_sink.hpp"
#include "srf/node/operators/muxer.hpp"
#include "srf/node/queue.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/pipeline/resources.hpp"
#include "srf/runnable/launch_options.hpp"
#include "srf/runnable/launchable.hpp"
#include "srf/runnable/types.hpp"
#include "srf/types.hpp"

#include <memory>
#include <unordered_map>

namespace srf::manifold {

namespace detail {

template <typename T>
class Balancer : public node::GenericSink<T>
{
  public:
    Balancer(RoundRobinEgress<T>& state) : m_state(state) {}

  private:
    void on_data(T&& data) final
    {
        // LOG(INFO) << "balancer node: " << data;
        m_state.await_write(std::move(data));
    }

    void will_complete() final
    {
        DVLOG(10) << "shutdown load-balancer - clear output channels";
        m_state.clear();
    };

    RoundRobinEgress<T>& m_state;
};

}  // namespace detail

template <typename T>
class LoadBalancer : public CompositeManifold<MuxedIngress<T>, QueueEgress<T>>
{
    using base_t = CompositeManifold<MuxedIngress<T>, QueueEgress<T>>;

  public:
    LoadBalancer(PortName port_name, pipeline::Resources& resources) : base_t(std::move(port_name), resources)
    {
        this->resources()
            .main()
            .enqueue([this] { node::make_edge(this->ingress().source(), this->egress().queue()); })
            .get();
    }

    void start() final {}

    void join() final {}
};

}  // namespace srf::manifold
