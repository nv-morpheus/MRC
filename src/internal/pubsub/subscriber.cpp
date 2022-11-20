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

#include "internal/pubsub/subscriber.hpp"

#include "internal/data_plane/resources.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/remote_descriptor/manager.hpp"

#include "srf/node/edge_builder.hpp"
#include "srf/node/rx_node.hpp"
#include "srf/protos/codable.pb.h"

namespace srf::internal::pubsub {

Subscriber::Subscriber(std::string service_name, runtime::Partition& runtime) : Base(std::move(service_name), runtime)
{}

void Subscriber::do_subscription_service_setup()
{
    CHECK_NE(tag(), 0);

    // reg
    auto& network_source = resources().network()->data_plane().server().deserialize_source().source(tag());

    auto network_handler = std::make_unique<srf::node::RxNode<memory::TransientBuffer, srf::runtime::RemoteDescriptor>>(
        rxcpp::operators::map([this](memory::TransientBuffer buffer) -> srf::runtime::RemoteDescriptor {
            return this->network_handler(buffer);
        }));

    srf::node::make_edge(network_source, *network_handler);
    srf::node::make_edge(*network_handler, *this);

    m_network_handler =
        resources().runnable().launch_control().prepare_launcher(std::move(network_handler))->ignition();

    m_network_handler->await_live();
}

void Subscriber::do_subscription_service_teardown()
{
    // disconnect from the deserialize source router
    // this will create a cascading shutdown
    resources().network()->data_plane().server().deserialize_source().drop_edge(tag());
}

void Subscriber::do_subscription_service_join()
{
    m_network_handler->await_join();
}

srf::runtime::RemoteDescriptor Subscriber::network_handler(memory::TransientBuffer& buffer)
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

const std::string& Subscriber::role() const
{
    return role_subscriber();
}
const std::set<std::string>& Subscriber::subscribe_to_roles() const
{
    static std::set<std::string> r;
    return r;
}
void Subscriber::update_tagged_instances(const std::string& role,
                                         const std::unordered_map<std::uint64_t, InstanceID>& tagged_instances)
{
    LOG(FATAL) << "subscribers should not receive updates";
}
}  // namespace srf::internal::pubsub
