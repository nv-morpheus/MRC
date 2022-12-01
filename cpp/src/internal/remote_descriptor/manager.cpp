/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/remote_descriptor/manager.hpp"

#include "internal/codable/codable_storage.hpp"
#include "internal/data_plane/client.hpp"
#include "internal/data_plane/request.hpp"
#include "internal/data_plane/resources.hpp"
#include "internal/network/resources.hpp"
#include "internal/remote_descriptor/decodable_storage.hpp"
#include "internal/remote_descriptor/storage.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/resources.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/codable/api.hpp"
#include "mrc/codable/encoded_object.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/protos/codable.pb.h"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>

#include <atomic>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace mrc::internal::remote_descriptor {

namespace {

ucs_status_t active_message_callback(
    void* arg, const void* header, size_t header_length, void* data, size_t length, const ucp_am_recv_param_t* param)
{
    DCHECK_EQ(header_length, sizeof(RemoteDescriptorDecrementMessage));

    const auto* const_msg   = static_cast<const RemoteDescriptorDecrementMessage*>(header);
    auto* decrement_channel = static_cast<node::SourceChannelWriteable<RemoteDescriptorDecrementMessage>*>(arg);

    // make a copy of the message and write it to the channel
    auto msg = *const_msg;
    CHECK(decrement_channel->await_write(std::move(msg)) == channel::Status::success);

    // we are done and data will not be used
    return UCS_OK;
}

}  // namespace

Manager::Manager(const InstanceID& instance_id, resources::PartitionResources& resources) :
  m_instance_id(instance_id),
  m_resources(resources)
{
    service_set_description(MRC_CONCAT_STR("mrc::remote_description_manager[" << instance_id << "]"));
    service_start();
    service_await_live();
}

Manager::~Manager()
{
    Service::call_in_destructor();
}

mrc::runtime::RemoteDescriptor Manager::make_remote_descriptor(mrc::codable::protos::RemoteDescriptor&& proto)
{
    // attach the resources for the partition in which this manager is operating to the rd's protobuf
    auto handle = std::make_unique<DecodableStorage>(std::move(proto), m_resources);
    return {shared_from_this(), std::move(handle)};
}

mrc::runtime::RemoteDescriptor Manager::register_encoded_object(std::unique_ptr<mrc::codable::EncodedStorage> object)
{
    CHECK(object);

    auto object_id = reinterpret_cast<std::size_t>(object.get());

    Storage storage(std::move(object));
    mrc::codable::protos::RemoteDescriptor rd;

    DVLOG(10) << "storing object_id: " << object_id << " with " << storage.tokens_count() << " tokens";

    rd.set_instance_id(m_instance_id);
    rd.set_object_id(object_id);
    rd.set_tokens(storage.tokens_count());
    *(rd.mutable_encoded_object()) = storage.encoding().proto();  // copy the proto::EncodedObject

    {
        // lock when modifying the map
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        auto search = m_stored_objects.find(object_id);
        CHECK(search == m_stored_objects.end());
        m_stored_objects.emplace(object_id, std::move(storage));
    }

    return make_remote_descriptor(std::move(rd));
}

std::unique_ptr<mrc::codable::ICodableStorage> Manager::create_storage()
{
    return std::make_unique<codable::CodableStorage>(m_resources);
}

std::size_t Manager::size() const
{
    return m_stored_objects.size();
}

void Manager::release_handle(std::unique_ptr<mrc::runtime::IRemoteDescriptorHandle> handle)
{
    CHECK(handle);
    const auto& rd = handle->remote_descriptor_proto();

    if (rd.instance_id() == m_instance_id)
    {
        decrement_tokens(rd.object_id(), rd.tokens());
    }
    else
    {
        // issue active message to remote instance_id to decrement tokens on remote object_id
        RemoteDescriptorDecrementMessage msg;
        msg.object_id = rd.object_id();
        msg.tokens    = rd.tokens();
        auto endpoint = m_resources.network()->data_plane().client().endpoint_shared(rd.instance_id());

        data_plane::Request request;
        data_plane::Client::async_am_send(active_message_id(), &msg, sizeof(msg), *endpoint, request);
        CHECK(request.await_complete());
    }
}

void Manager::decrement_tokens(std::size_t object_id, std::size_t token_count)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    LOG(INFO) << "decrementing " << token_count << " tokens from object_id: " << object_id;
    DVLOG(10) << "decrementing " << token_count << " tokens from object_id: " << object_id;
    auto search = m_stored_objects.find(object_id);
    CHECK(search != m_stored_objects.end());
    auto remaining = search->second.decrement_tokens(token_count);
    if (remaining == 0)
    {
        DVLOG(10) << "destroying object_id: " << object_id;
        m_stored_objects.erase(search);
    }
}

void Manager::do_service_start()
{
    m_decrement_channel    = std::make_unique<node::SourceChannelWriteable<RemoteDescriptorDecrementMessage>>();
    auto decrement_handler = std::make_unique<node::RxSink<RemoteDescriptorDecrementMessage>>(
        [this](RemoteDescriptorDecrementMessage msg) { decrement_tokens(msg.object_id, msg.tokens); });
    decrement_handler->update_channel(
        std::make_unique<channel::BufferedChannel<RemoteDescriptorDecrementMessage>>(128));
    node::make_edge(*m_decrement_channel, *decrement_handler);

    mrc::runnable::LaunchOptions launch_options;
    launch_options.engine_factory_name = "main";

    m_decrement_handler = m_resources.runnable()
                              .launch_control()
                              .prepare_launcher(launch_options, std::move(decrement_handler))
                              ->ignition();

    // register active message handler
    ucp_am_handler_param params;
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                        UCP_AM_HANDLER_PARAM_FIELD_CB | UCP_AM_HANDLER_PARAM_FIELD_ARG;
    params.id    = active_message_id();
    params.flags = UCP_AM_FLAG_WHOLE_MSG;
    params.cb    = active_message_callback;
    params.arg   = m_decrement_channel.get();

    CHECK_EQ(ucp_worker_set_am_recv_handler(m_resources.network()->ucx().worker().handle(), &params), UCS_OK);
}

void Manager::do_service_stop()
{
    while (!m_stored_objects.empty())
    {
        LOG_EVERY_N(WARNING, INT32_MAX) << "awaiting release of remote descriptors";  // NOLINT
        boost::this_fiber::yield();
    }

    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    DCHECK_EQ(m_stored_objects.size(), 0);

    CHECK(m_decrement_channel);
    CHECK(m_decrement_handler);

    // deregister active message handler
    ucp_am_handler_param params;
    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID | UCP_AM_HANDLER_PARAM_FIELD_CB;
    params.id         = active_message_id();
    params.cb         = nullptr;
    CHECK_EQ(ucp_worker_set_am_recv_handler(m_resources.network()->ucx().worker().handle(), &params), UCS_OK);

    // close channel
    m_decrement_channel.reset();
}
void Manager::do_service_kill()
{
    do_service_stop();
}
void Manager::do_service_await_live()
{
    CHECK(m_decrement_handler);
    m_decrement_handler->await_live();
}
void Manager::do_service_await_join()
{
    CHECK(m_decrement_handler);
    m_decrement_handler->await_join();
}
std::uint32_t Manager::active_message_id()
{
    return 10000;
}

const mrc::codable::IDecodableStorage& Manager::encoding(const std::size_t& object_id) const
{
    std::lock_guard lock(m_mutex);
    auto search = m_stored_objects.find(object_id);
    CHECK(search != m_stored_objects.end());
    return search->second.encoding();
}
InstanceID Manager::instance_id() const
{
    return m_instance_id;
}
mrc::runtime::RemoteDescriptor Manager::make_remote_descriptor(
    std::unique_ptr<mrc::runtime::IRemoteDescriptorHandle> handle)
{
    return {shared_from_this(), std::move(handle)};
}
}  // namespace mrc::internal::remote_descriptor
