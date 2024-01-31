/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/data_plane_manager.hpp"

#include "internal/control_plane/client.hpp"
#include "internal/data_plane/data_plane_resources.hpp"
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/future/future.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace mrc::edge {
template <typename T>
class IWritableProvider;
}  // namespace mrc::edge

namespace mrc::runtime {

DataPlaneSystemManager::DataPlaneSystemManager(IInternalRuntimeProvider& runtime) :
  AsyncService(MRC_CONCAT_STR("DataPlaneSystemManager")),
  InternalRuntimeProvider(runtime)
{}

DataPlaneSystemManager::~DataPlaneSystemManager()
{
    AsyncService::call_in_destructor();
}

std::shared_ptr<edge::IReadableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_readable_ingress_channel(InstanceID port_address) const
{
    std::unique_lock lock(m_port_mutex);

    if (m_ingress_port_channels.contains(port_address))
    {
        // Now check that its alive otherwise we fall through
        if (auto port = m_ingress_port_channels.at(port_address).lock())
        {
            return port;
        }
    }

    auto* mutable_this = const_cast<DataPlaneSystemManager*>(this);

    auto port_channel = std::make_shared<node::Queue<std::unique_ptr<ValueDescriptor>>>();

    mutable_this->m_ingress_port_channels[port_address] = port_channel;

    return port_channel;
}

std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_writable_ingress_channel(InstanceID port_address) const
{
    std::unique_lock lock(m_port_mutex);

    if (m_ingress_port_channels.contains(port_address))
    {
        // Now check that its alive otherwise we fall through
        if (auto port = m_ingress_port_channels.at(port_address).lock())
        {
            return port;
        }
    }

    auto* mutable_this = const_cast<DataPlaneSystemManager*>(this);

    auto port_channel = std::make_shared<node::Queue<std::unique_ptr<ValueDescriptor>>>();

    mutable_this->m_ingress_port_channels[port_address] = port_channel;

    return port_channel;
}

std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_writable_egress_channel(InstanceID port_address) const
{
    std::unique_lock lock(m_port_mutex);

    if (m_egress_port_channels.contains(port_address))
    {
        // Now check that its alive otherwise we fall through
        if (auto port = m_egress_port_channels.at(port_address).lock())
        {
            return port;
        }
    }

    auto* mutable_this = const_cast<DataPlaneSystemManager*>(this);

    auto endpoint = m_resources->find_endpoint(port_address);

    auto port_channel = std::make_shared<node::LambdaSinkComponent<std::unique_ptr<ValueDescriptor>>>(
        [mutable_this, endpoint](std::unique_ptr<ValueDescriptor>&& data) {
            return mutable_this->send_descriptor(endpoint, std::move(data));
        });

    mutable_this->m_egress_port_channels[port_address] = port_channel;

    return port_channel;
}

std::shared_ptr<node::Queue<std::unique_ptr<Descriptor>>> DataPlaneSystemManager::get_incoming_port_channel(
    InstanceID port_address) const
{
    std::unique_lock lock(m_port_mutex);

    if (m_incoming_port_channels.contains(port_address))
    {
        // Now check that its alive otherwise we fall through
        if (auto port = m_incoming_port_channels.at(port_address).lock())
        {
            return port;
        }
    }

    auto* mutable_this = const_cast<DataPlaneSystemManager*>(this);

    auto port_channel = std::make_shared<node::Queue<std::unique_ptr<Descriptor>>>();

    mutable_this->m_incoming_port_channels[port_address] = port_channel;

    return port_channel;
}

std::shared_ptr<edge::IWritableProvider<std::unique_ptr<Descriptor>>> DataPlaneSystemManager::get_outgoing_port_channel(
    InstanceID port_address) const
{
    std::unique_lock lock(m_port_mutex);

    if (m_outgoing_port_channels.contains(port_address))
    {
        // Now check that its alive otherwise we fall through
        if (auto port = m_outgoing_port_channels.at(port_address).lock())
        {
            return port;
        }
    }

    auto* mutable_this = const_cast<DataPlaneSystemManager*>(this);

    auto port_channel = std::make_shared<node::Queue<std::unique_ptr<Descriptor>>>();

    mutable_this->m_outgoing_port_channels[port_address] = port_channel;

    return port_channel;
}

void DataPlaneSystemManager::do_service_start(std::stop_token stop_token)
{
    m_resources = std::make_unique<data_plane::DataPlaneResources2>();

    Promise<void> completed_promise;

    // Block until we get a state update with this worker
    this->runtime().control_plane().state_update_obs().subscribe(
        [this](auto state) {
            this->process_state_update(state);
        },
        [this, &completed_promise](std::exception_ptr ex_ptr) {
            try
            {
                std::rethrow_exception(ex_ptr);
            } catch (const std::exception& ex)
            {
                LOG(ERROR) << this->debug_prefix() << " Error in subscription. Message: " << ex.what();
            }

            this->service_kill();

            // Must call the completed promise
            completed_promise.set_value();
        },
        [&completed_promise] {
            completed_promise.set_value();
        });

    this->mark_started();

    completed_promise.get_future().get();
}

void DataPlaneSystemManager::process_state_update(const control_plane::state::ControlPlaneState& state)
{
    // Ensure that endpoints exist for all workers
    for (const auto& [worker_id, worker] : state.workers())
    {
        if (!m_resources->has_endpoint(worker.ucx_address()))
        {
            // Create endpoints for all other available workers
            m_resources->create_endpoint(worker.ucx_address(), worker.executor_id());
        }
    }
}

channel::Status DataPlaneSystemManager::send_descriptor(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                        std::unique_ptr<ValueDescriptor>&& descriptor)
{
    auto block_provider = std::make_shared<memory::memory_block_provider>();

    // Convert from value to local descriptor
    auto local = LocalDescriptor2::from_value(std::move(descriptor), block_provider);

    // Convert from local to remote descriptor
    auto remote = RemoteDescriptor2::from_local(std::move(local), *m_resources);

    // Serialize to bytes
    auto serialized_buffer = remote->to_bytes(memory::malloc_memory_resource::instance());

    // Save the remote block to the memory manager

    // Send the bytes to the remote worker
    auto request = m_resources->am_send_async(endpoint, serialized_buffer);

    m_resources->wait_requests(std::vector<std::shared_ptr<ucxx::Request>>{request});

    return channel::Status::success;
}

DataPlaneManager::DataPlaneManager(IInternalRuntimeProvider& runtime, size_t partition_id) :
  AsyncService(MRC_CONCAT_STR("DataPlaneManager[" << partition_id << "]")),
  InternalRuntimeProvider(runtime)
{}

DataPlaneManager::~DataPlaneManager()
{
    AsyncService::call_in_destructor();
}

void DataPlaneManager::sync_state(const control_plane::state::Worker& worker) {}

void DataPlaneManager::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    std::stop_callback stop_callback(stop_token, [&completed_promise]() {
        completed_promise.set_value();
    });

    this->mark_started();

    completed_promise.get_future().get();
}

}  // namespace mrc::runtime
