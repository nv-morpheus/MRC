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
#include "mrc/edge/edge_builder.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/memory/memory_block_provider.hpp"
#include "mrc/memory/resources/host/malloc_memory_resource.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/queue.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <rxcpp/rx.hpp>
#include <ucs/type/status.h>
#include <ucxx/request.h>

#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
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

std::string DataPlaneSystemManager::get_ucx_address() const
{
    return m_resources->address();
}

std::shared_ptr<edge::IReadableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_readable_ingress_channel(PortAddress2 port_address) const
{
    return this->get_ingress_port_channel(port_address);
}

std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_writable_ingress_channel(PortAddress2 port_address) const
{
    return this->get_ingress_port_channel(port_address);
}

std::shared_ptr<edge::IWritableProvider<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::
    get_writable_egress_channel(PortAddress2 port_address) const
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

    auto port_channel = std::make_shared<node::LambdaSinkComponent<std::unique_ptr<ValueDescriptor>>>(
        [mutable_this, port_address](std::unique_ptr<ValueDescriptor>&& data) {
            return mutable_this->send_descriptor(port_address, std::move(data));
        },
        [mutable_this, port_address]() {
            // Stream has closed. Need to send the closed message for this port
            VLOG(10) << "Stream has closed for port " << port_address;
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

    auto resources_progress = std::make_unique<node::LambdaSource<int>>([this](rxcpp::subscriber<int>& subscriber) {
        // Pull from the resources channel
        while (subscriber.is_subscribed())
        {
            if (!this->m_resources->progress())
            {
                // TODO(MDD): Convert this to an exponential backoff
                boost::this_fiber::yield();
            }
        }
    });

    // Create a runnable to be pulling messages off the resources object
    auto pull_remote_descriptors =
        std::make_unique<node::LambdaSource<std::pair<PortAddress2, std::unique_ptr<ValueDescriptor>>>>(
            [this](rxcpp::subscriber<std::pair<PortAddress2, std::unique_ptr<ValueDescriptor>>>& subscriber) {
                // Pull from the resources channel
                while (subscriber.is_subscribed())
                {
                    std::unique_ptr<RemoteDescriptor2> temp_remote;

                    auto status = this->m_resources->get_inbound_channel().await_read_until(
                        temp_remote,
                        channel::clock_t::now() + std::chrono::milliseconds(100));

                    if (status == channel::Status::success)
                    {
                        auto destination = PortAddress2(temp_remote->encoded_object().destination_address());

                        // Convert from remote to local descriptor
                        auto local = LocalDescriptor2::from_remote(std::move(temp_remote), *m_resources);

                        subscriber.on_next(std::make_pair(destination, std::move(local)));
                    }
                }
            });

    m_inbound_dispatcher = std::make_unique<node::TaggedRouter<PortAddress2, std::unique_ptr<ValueDescriptor>>>();

    mrc::make_edge(*pull_remote_descriptors, *m_inbound_dispatcher);

    // Start the runnables
    this->child_runnable_start("remote_progress",
                               mrc::runnable::LaunchOptions("mrc_network", 1),
                               std::move(resources_progress));

    this->child_runnable_start("pull_remote_descriptors",
                               mrc::runnable::LaunchOptions("mrc_network", 1, 8),
                               std::move(pull_remote_descriptors));

    this->mark_started();

    completed_promise.get_future().get();
}

void DataPlaneSystemManager::process_state_update(const control_plane::state::ControlPlaneState& state)
{
    if (!m_resources->has_instance_id())
    {
        // Get the current connection ID and set it in the resources
        auto machine_id = this->runtime().control_plane().machine_id();

        m_resources->set_instance_id(machine_id);
    }

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

std::shared_ptr<node::Queue<std::unique_ptr<ValueDescriptor>>> DataPlaneSystemManager::get_ingress_port_channel(
    PortAddress2 port_address) const
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

    auto port_channel = std::shared_ptr<node::Queue<std::unique_ptr<ValueDescriptor>>>(
        new node::Queue<std::unique_ptr<ValueDescriptor>>(),
        [mutable_this, port_address](node::Queue<std::unique_ptr<ValueDescriptor>>* to_delete) {
            // Make sure to unregister before deleting
            mutable_this->m_inbound_dispatcher->drop_source(port_address);

            delete to_delete;
        });

    // Make a connection between the incoming channel and this port-specific channel
    mrc::make_edge(*m_inbound_dispatcher->get_source(port_address), *port_channel);

    mutable_this->m_ingress_port_channels[port_address] = port_channel;

    return port_channel;
}

channel::Status DataPlaneSystemManager::send_descriptor(PortAddress2 port_destination,
                                                        std::unique_ptr<ValueDescriptor> descriptor)
{
    VLOG(10) << "Sending descriptor to " << port_destination << "[starting]";

    auto block_provider = std::make_shared<memory::memory_block_provider>();

    // Convert from value to local descriptor
    auto local = LocalDescriptor2::from_value(std::move(descriptor), block_provider);

    // Convert from local to remote descriptor
    auto remote = RemoteDescriptor2::from_local(std::move(local), *m_resources);

    // TODO(MDD): Temp setting the source address as just the executor (so the tokens can be decremented)
    remote->encoded_object().set_source_address(
        PortAddress2(static_cast<uint16_t>(m_resources->get_instance_id()), 0, 0, 0).combined);

    // Set the destination
    remote->encoded_object().set_destination_address(port_destination.combined);

    // Serialize to bytes
    auto serialized_buffer = remote->to_bytes(memory::malloc_memory_resource::instance());

    // Get the endpoint for this message
    auto endpoint = m_resources->find_endpoint(port_destination.executor_id);

    Promise<void> completed_promise;

    auto completed_future = completed_promise.get_future();

    VLOG(10) << "Sending descriptor to " << port_destination << "[sending]";

    // Send the bytes to the remote worker
    auto request = m_resources->am_send_async(
        endpoint,
        serialized_buffer,
        ucxx::AmReceiverCallbackInfo("MRC", 1 << 2),
        ucxx::RequestCallbackUserFunction([&completed_promise](ucs_status_t status, std::shared_ptr<void>) {
            if (status != UCS_OK)
            {
                completed_promise.set_exception(
                    std::make_exception_ptr(std::runtime_error("Failed to send descriptor")));
            }
            else
            {
                completed_promise.set_value();
            }
        }));

    VLOG(10) << "Sending descriptor to " << port_destination << "[waiting]";

    // TODO(MDD): Do we need to wait for the request to send or can we just assume it sends?
    completed_future.get();
    // m_resources->wait_requests(std::vector<std::shared_ptr<ucxx::Request>>{request});

    VLOG(10) << "Sending descriptor to " << port_destination << "[complete]";

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
