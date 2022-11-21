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

#pragma once

#include "srf/codable/api.hpp"
#include "srf/pubsub/forward.hpp"

#include <cstdint>
#include <memory>

namespace srf::runtime {

class IPartition;
class IRemoteDescriptorManager;

/**
 * @brief Top-level interface for public runtime resources/components
 *
 * The SRF runtime constructs a set of one more partitions based on user configuration options. Based on the PCIe
 * topology of the machine, partitions are a set of topologically aligned resources. There will be at least as many
 * partitions as there are accessiable NVIDIA GPU compute devices such that partitions [0, gpu_count()] are arranged in
 * CUDA device_id order. Based on user options, each partition will consist of a set of CPU cores, memory allocators, an
 * optional GPU and an optional UCX context initialized on the NIC "closest" to the GPU/NUMA node.
 *
 * Most resources are specific to a partition. Each Runnable's Context will have a runtime discoverable default
 * partition id which is the suggestion set of resources to be used with that runnable.
 */
class IRuntime
{
  public:
    virtual ~IRuntime() = default;

    virtual std::size_t partition_count() const = 0;
    virtual std::size_t gpu_count() const       = 0;

    virtual IPartition& partition(std::size_t partition_id) = 0;
};

/**
 * @brief Partition-level interface for publically exposed resources/components
 */
class IPartition
{
  public:
    virtual ~IPartition() = default;

    virtual IRemoteDescriptorManager& remote_descriptor_manager() = 0;

    virtual std::unique_ptr<codable::ICodableStorage> make_codable_storage() = 0;

  private:
    virtual std::shared_ptr<pubsub::IPublisherService> make_publisher_service(
        const std::string& name, const pubsub::PublisherPolicy& policy) = 0;

    virtual std::shared_ptr<pubsub::ISubscriberService> make_subscriber_service(const std::string& name) = 0;

    // enable access to private methods

    template <typename T>
    friend class pubsub::Publisher;

    template <typename T>
    friend class pubsub::Subscriber;
};

}  // namespace srf::runtime
