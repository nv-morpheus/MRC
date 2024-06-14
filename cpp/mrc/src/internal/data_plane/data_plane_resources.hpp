/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/data_plane/request.hpp"
#include "internal/memory/transient_pool.hpp"
#include "internal/resources/partition_resources_base.hpp"
#include "internal/service.hpp"
#include "internal/ucx/forward.hpp"

#include "mrc/channel/channel.hpp"
#include "mrc/memory/buffer_view.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runtime/remote_descriptor.hpp"
#include "mrc/types.hpp"

#include <ucp/api/ucp_def.h>
#include <ucxx/typedefs.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

// using ucp_rkey_h        = struct ucp_rkey*;
// using ucs_memory_type_t = enum ucs_memory_type;

namespace ucxx {
class Context;
class Endpoint;
class Worker;
class Request;
class Address;
}  // namespace ucxx

namespace mrc::control_plane {
class Client;
}  // namespace mrc::control_plane
namespace mrc::memory {
class HostResources;
}  // namespace mrc::memory
namespace mrc::network {
class NetworkResources;
}  // namespace mrc::network
namespace mrc::ucx {
class RegistrationCache;
class RegistrationCache2;
class UcxResources;
}  // namespace mrc::ucx

namespace mrc::node {
template <typename T>
class Queue;
}  // namespace mrc::node

namespace mrc::data_plane {
class Client;
class Server;

/**
 * @brief ArchitectResources hold and is responsible for constructing any object that depending the UCX data plane
 *
 */
class DataPlaneResources final : private Service, private resources::PartitionResourceBase
{
  public:
    DataPlaneResources(resources::PartitionResourceBase& base,
                       ucx::UcxResources& ucx,
                       memory::HostResources& host,
                       const InstanceID& instance_id,
                       control_plane::Client& control_plane_client);
    ~DataPlaneResources() final;

    Client& client();
    Server& server();

    const InstanceID& instance_id() const;
    std::string ucx_address() const;
    const ucx::RegistrationCache& registration_cache() const;

    static mrc::runnable::LaunchOptions launch_options(std::size_t concurrency);

  private:
    void do_service_start() final;
    void do_service_await_live() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_join() final;

    ucx::UcxResources& m_ucx;
    memory::HostResources& m_host;
    control_plane::Client& m_control_plane_client;
    InstanceID m_instance_id;

    memory::TransientPool m_transient_pool;
    std::unique_ptr<Server> m_server;
    std::unique_ptr<Client> m_client;

    friend network::NetworkResources;
};

class DataPlaneResources2
{
  public:
    DataPlaneResources2();
    virtual ~DataPlaneResources2();

    void set_instance_id(uint64_t instance_id);
    bool has_instance_id() const;
    uint64_t get_instance_id() const;

    ucxx::Context& context() const;

    ucxx::Worker& worker() const;

    std::string address() const;

    ucx::RegistrationCache2& registration_cache() const;

    std::shared_ptr<ucxx::Endpoint> create_endpoint(const std::string& address, uint64_t instance_id);

    bool has_endpoint(const std::string& address) const;
    std::shared_ptr<ucxx::Endpoint> find_endpoint(const std::string& address) const;
    std::shared_ptr<ucxx::Endpoint> find_endpoint(uint64_t instance_id) const;

    // Advances the worker
    bool progress();

    // Flushes the worker
    bool flush();

    // Wait for the requests to complete
    void wait_requests(const std::vector<std::shared_ptr<ucxx::Request>>& requests);

    std::shared_ptr<ucxx::Request> memory_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     memory::const_buffer_view buffer_view,
                                                     uintptr_t remote_addr,
                                                     ucp_rkey_h rkey,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> memory_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     const void* addr,
                                                     std::size_t bytes,
                                                     uintptr_t remote_addr,
                                                     ucp_rkey_h rkey,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> memory_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     memory::buffer_view buffer_view,
                                                     uintptr_t remote_addr,
                                                     const void* packed_rkey_data,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> memory_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     void* addr,
                                                     std::size_t bytes,
                                                     uintptr_t remote_addr,
                                                     const void* packed_rkey_data,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     memory::const_buffer_view buffer_view,
                                                     uint64_t tag,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> tagged_send_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     const void* buffer,
                                                     size_t bytes,
                                                     uint64_t tag,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> tagged_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint,
                                                     void* buffer,
                                                     size_t bytes,
                                                     uint64_t tag,
                                                     uint64_t tag_mask,
                                                     ucxx::RequestCallbackUserFunction callback_function = nullptr,
                                                     ucxx::RequestCallbackUserData callback_data         = nullptr);

    std::shared_ptr<ucxx::Request> am_send_async(
        std::shared_ptr<ucxx::Endpoint> endpoint,
        memory::const_buffer_view buffer_view,
        std::optional<ucxx::AmReceiverCallbackInfo> callback_info = std::nullopt,
        ucxx::RequestCallbackUserFunction callback_function       = nullptr,
        ucxx::RequestCallbackUserData callback_data               = nullptr);

    std::shared_ptr<ucxx::Request> am_send_async(
        std::shared_ptr<ucxx::Endpoint> endpoint,
        const void* addr,
        std::size_t bytes,
        ucs_memory_type_t mem_type,
        std::optional<ucxx::AmReceiverCallbackInfo> callback_info = std::nullopt,
        ucxx::RequestCallbackUserFunction callback_function       = nullptr,
        ucxx::RequestCallbackUserData callback_data               = nullptr);
    std::shared_ptr<ucxx::Request> am_recv_async(std::shared_ptr<ucxx::Endpoint> endpoint);

    uint64_t register_remote_decriptor(std::shared_ptr<runtime::RemoteDescriptorImpl2> remote_descriptor);

    channel::Egress<std::unique_ptr<runtime::RemoteDescriptor2>>& get_inbound_channel() const;

  private:
    virtual std::shared_ptr<runtime::RemoteDescriptorImpl2> get_descriptor(uint64_t object_id);

    std::optional<uint64_t> m_instance_id;  // Global ID used to identify this instance

    std::shared_ptr<ucxx::Context> m_context;
    std::shared_ptr<ucxx::Worker> m_worker;
    std::shared_ptr<ucxx::Address> m_address;

    std::shared_ptr<ucx::RegistrationCache2> m_registration_cache;

    std::map<std::string, std::shared_ptr<ucxx::Endpoint>> m_endpoints_by_address;
    std::map<uint64_t, std::shared_ptr<ucxx::Endpoint>> m_endpoints_by_id;

    std::atomic_size_t m_next_object_id{0};

    std::unique_ptr<BufferedChannel<std::unique_ptr<runtime::RemoteDescriptor2>>> m_inbound_channel;

    mutable Mutex m_mutex;

    // std::shared_ptr<node::Queue<std::unique_ptr<runtime::ValueDescriptor>>> m_outbound_descriptors;
    // std::map<InstanceID, std::weak_ptr<node::Queue<std::unique_ptr<runtime::ValueDescriptor>>>>
    // m_inbound_port_channels;

    uint64_t get_next_object_id();

  protected:
    std::map<uint64_t, std::shared_ptr<runtime::RemoteDescriptorImpl2>> m_remote_descriptor_by_id;
};

}  // namespace mrc::data_plane
