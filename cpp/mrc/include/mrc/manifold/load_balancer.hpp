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

#include "mrc/core/addresses.hpp"
#include "mrc/manifold/composite_manifold.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/pipeline/resources.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/types.hpp"

#include <memory>
#include <unordered_map>

namespace mrc::manifold {

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
class LoadBalancer : public CompositeManifold<MuxedIngress<T>, RoundRobinEgress<T>>
{
    using base_t = CompositeManifold<MuxedIngress<T>, RoundRobinEgress<T>>;

  public:
    LoadBalancer(PortName port_name, pipeline::Resources& resources) : base_t(std::move(port_name), resources)
    {
        m_launch_options.engine_factory_name = "main";
        m_launch_options.pe_count            = 1;
        m_launch_options.engines_per_pe      = 8;

        // construct any resources
        this->resources()
            .main()
            .enqueue([this] {
                m_balancer = std::make_unique<detail::Balancer<T>>(this->egress());
                node::make_edge(this->ingress().source(), *m_balancer);
            })
            .get();
    }

    void start() final
    {
        this->resources()
            .main()
            .enqueue([this] {
                if (m_runner)
                {
                    // todo(#179) - validate this fix and improve test coverage
                    // this will be handled now by the default behavior of SourceChannel::no_channel method
                    // CHECK(!this->egress().output_channels().empty()) << "no egress channels on manifold";
                    return;
                }
                CHECK(m_balancer);
                m_runner = this->resources()
                               .launch_control()
                               .prepare_launcher(launch_options(), std::move(m_balancer))
                               ->ignition();
            })
            .get();
    }

    void join() final
    {
        m_runner->await_join();
    }

    const runnable::LaunchOptions& launch_options() const
    {
        return m_launch_options;
    }

  private:
    // launch options
    runnable::LaunchOptions m_launch_options;

    // this is the progress engine that will drive the load balancer
    std::unique_ptr<node::GenericSink<T>> m_balancer;

    // runner
    std::unique_ptr<runnable::Runner> m_runner{nullptr};
};

}  // namespace mrc::manifold
