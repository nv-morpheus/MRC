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
#include "internal/runtime/runtime_provider.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/exceptions/runtime_error.hpp"
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
    // m_previous_state = state;
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
