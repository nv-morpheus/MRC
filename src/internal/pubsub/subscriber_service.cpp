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

#include "internal/pubsub/subscriber_service.hpp"

#include "rxcpp/operators/rx-map.hpp"
#include "rxcpp/sources/rx-iterate.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/data_plane/server.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/network/resources.hpp"
#include "internal/remote_descriptor/manager.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/resources.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/node/operators/router.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/runnable/launch_control.hpp"
#include "srf/runnable/launcher.hpp"
#include "srf/utils/bytes_to_string.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <optional>
#include <ostream>
#include <vector>

namespace srf::internal::pubsub {

SubscriberService::SubscriberService(std::string service_name, runtime::Partition& runtime) :
  Base(std::move(service_name), runtime)
{}

void SubscriberService::do_subscription_service_setup()
{
    CHECK_NE(tag(), 0);

    // reg
    auto& network_source = resources().network()->data_plane().server().deserialize_source().source(tag());

    auto network_handler = std::make_unique<srf::node::RxNode<memory::TransientBuffer, srf::runtime::RemoteDescriptor>>(
        rxcpp::operators::map([this](memory::TransientBuffer buffer) -> srf::runtime::RemoteDescriptor {
            return this->network_handler(buffer);
        }));

    DVLOG(10) << "form edge:  network_soruce -> network_handler";
    srf::node::make_edge(network_source, *network_handler);

    DVLOG(10) << "form edge:  network_handler -> rd_channel (ISubscriberService::SourceChannelWriteable)";
    srf::node::make_edge(*network_handler, *this);

    DVLOG(10) << "starting network handler node";
    m_network_handler =
        resources().runnable().launch_control().prepare_launcher(std::move(network_handler))->ignition();

    m_network_handler->await_live();
    DVLOG(10) << "finished internal:pubsub::SubscriberService setup";
}

void SubscriberService::do_subscription_service_teardown()
{
    // disconnect from the deserialize source router
    // this will create a cascading shutdown
    resources().network()->data_plane().server().deserialize_source().drop_edge(tag());
}

void SubscriberService::do_subscription_service_join()
{
    m_network_handler->await_join();
}

srf::runtime::RemoteDescriptor SubscriberService::network_handler(memory::TransientBuffer& buffer)
{
    DVLOG(10) << "transient buffer holding the rd: " << srf::bytes_to_string(buffer.bytes());

    // deserialize remote descriptor handle/proto from transient buffer
    srf::codable::protos::RemoteDescriptor proto;
    CHECK(proto.ParseFromArray(buffer.data(), buffer.bytes()));

    // release transient buffer so it can be reused
    buffer.release();

    // create a remote descriptor via the local RD manager taking ownership of the handle
    return runtime().remote_descriptor_manager().make_remote_descriptor(std::move(proto));
}

const std::string& SubscriberService::role() const
{
    return role_subscriber();
}
const std::set<std::string>& SubscriberService::subscribe_to_roles() const
{
    static std::set<std::string> r;
    return r;
}
void SubscriberService::update_tagged_instances(const std::string& role,
                                                const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances)
{
    LOG(FATAL) << "subscribers should not receive updates";
}
}  // namespace srf::internal::pubsub
