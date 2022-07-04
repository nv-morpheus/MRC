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

#include "internal/resources/forward.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"
#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/registration_cache.hpp"
#include "internal/ucx/worker.hpp"

#include "srf/channel/status.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/node/generic_source.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/runnable/context.hpp"
#include "srf/runnable/launch_control.hpp"
#include "srf/runnable/runner.hpp"
#include "srf/types.hpp"

#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <ucp/api/ucp_def.h>

#include <memory>
#include <utility>

// this node gets ucx tagged messages from ucp_tag_probe_nb
// events which do not required a recv get pushed immediately to their downstream
// sinks, while message that do require can be optionally handled on the same
// active fiber that gets the probed message.
//
// for now, we will treat messages that require a matching recv to be of
// high-enough priority where we do not want to delay the issuing of the recv
// on the other side of a channel.
//
// if in time, we wish to have the network recv/get calls backoff, we will
// simply create a downstream recv node and issue the probed msg data to the
// recv node via a buffered channel

// todo: determine if moving the allocators to downstream object might be a
// a better abstraction. this means the recv would need to be completed in
// the downstream object. downstream does not necessarily imply that we will
// issue a channel write to a sink, rather we do a lookup on the destination
// channel from a map, then pass the msg/msg_info to the interface for that
// channel decoder, which should issue the recv in the same context as was
// called and return immediately.

namespace srf::internal::data_plane {

namespace detail {
struct PrePostedRecvInfo
{
    ucp_worker_h worker;
    node::SourceChannelWriteable<ucp_tag_t>* channel;
    void* request;
    // std::array<std::byte, 2048> buffer;
};
}  // namespace detail

using network_event_t = std::pair<PortAddress, srf::memory::buffer_view>;

class Server final : public Service, public resources::PartitionResourceBase
{
  public:
    Server(resources::PartitionResourceBase& provider, ucx::Resources& ucx, memory::HostResources& host);
    ~Server() final;

    ucx::WorkerAddress worker_address() const;

    node::Router<PortAddress, srf::memory::buffer_view>& deserialize_source();

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    const std::size_t m_pre_posted_recv_count{16};

    // ucx resources
    ucx::Resources& m_ucx;
    memory::HostResources& m_host;

    // deserialization nodes will connect to this source wtih their port id
    // the source for this router is the private GenericSoruce of this object
    std::shared_ptr<node::Router<PortAddress, srf::memory::buffer_view>> m_deserialize_source;

    // the remote descriptor manager will connect to this source
    // data will be emitted on this source as a conditional branch of data source
    std::unique_ptr<node::SourceChannelWriteable<ucp_tag_t>> m_rd_source;

    // pre-posted recv state
    std::vector<detail::PrePostedRecvInfo> m_pre_posted_recv_info;

    // runner for the ucx progress engine event source
    std::unique_ptr<srf::runnable::Runner> m_progress_engine;
};

}  // namespace srf::internal::data_plane
