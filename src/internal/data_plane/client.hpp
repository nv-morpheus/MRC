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

#include "internal/service.hpp"

#include <srf/protos/remote_descriptor.pb.h>
#include <srf/channel/status.hpp>
#include <srf/codable/encoded_object.hpp>
#include <srf/node/source_channel.hpp>
#include <srf/runnable/launch_control.hpp>
#include <srf/runnable/runner.hpp>
#include <srf/types.hpp>
#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/worker.hpp"

#include <ucp/api/ucp_def.h>
#include <rxcpp/rx.hpp>  // IWYU pragma: keep

#include <cstddef>
#include <map>
#include <memory>

namespace srf::internal::data_plane {

// todo(ryan) - rename NetworkSendManager -> DataPlaneAPI

class Client final : public Service
{
  public:
    Client(std::shared_ptr<ucx::Context> context);
    ~Client() final;

    /**
     * @brief Register a UCX Worker address with an InstanceID
     */
    void register_instance(InstanceID instance_id, ucx::WorkerAddress worker_address);

    /**
     * @brief Send an EncodedObject to the PortAddress at InstanceID
     *
     * @note Issue #122 would elmininate the need for InstanceID to be passed; however, this might also be an
     * optimization path to avoid a second PortAddress to InstanceID lookup. NetworkSendManager should issue
     * the await_send with only port_address and encoded_object; however, the internal should be able to short
     * circuit the translation.
     *
     * @param instance_id
     * @param port_address
     * @param encoded_object
     */
    void await_send(const InstanceID& instance_id,
                    const PortAddress& port_address,
                    const codable::EncodedObject& encoded_object);

    // number of established remote instances
    std::size_t connections() const;

    // determine if connected to a given remote instance
    bool is_connected_to(InstanceID) const;

    void decrement_remote_descriptor(InstanceID, ObjectID);

    // void get(const protos::RemoteDescriptor&, void*, size_t);
    // void get(const protos::RemoteDescriptor&, Descriptor&);

  protected:
    // issue tag only send - no payload data
    void issue_network_event(InstanceID, ucp_tag_t);

    // get endpoint for instance id
    const ucx::Endpoint& endpoint(InstanceID) const;

    void push_request(void* request);

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    std::shared_ptr<ucx::Worker> m_worker;
    std::unique_ptr<node::SourceChannelWriteable<void*>> m_ucx_request_channel;
    std::unique_ptr<runnable::Runner> m_progress_engine;

    std::map<InstanceID, ucx::WorkerAddress> m_workers;
    mutable std::map<InstanceID, std::shared_ptr<ucx::Endpoint>> m_endpoints;
};

}  // namespace srf::internal::data_plane
