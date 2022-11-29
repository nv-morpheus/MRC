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

#include "internal/codable/decodable_storage_view.hpp"
#include "internal/codable/storage_view.hpp"
#include "internal/resources/forward.hpp"

#include "srf/protos/codable.pb.h"
#include "srf/runtime/remote_descriptor_handle.hpp"
#include "srf/utils/macros.hpp"

namespace srf::internal::remote_descriptor {

class DecodableStorage final : public codable::DecodableStorageView,
                               public codable::StorageView,
                               public srf::runtime::IRemoteDescriptorHandle
{
  public:
    DecodableStorage(srf::codable::protos::RemoteDescriptor&& proto, resources::PartitionResources& resources);
    ~DecodableStorage() final = default;

    DELETE_COPYABILITY(DecodableStorage);
    DELETE_MOVEABILITY(DecodableStorage);

    const srf::codable::protos::RemoteDescriptor& remote_descriptor_proto() const final;

  protected:
    const srf::codable::protos::EncodedObject& get_proto() const final;

    resources::PartitionResources& resources() const final;

  private:
    srf::codable::protos::RemoteDescriptor m_proto;
    resources::PartitionResources& m_resources;
};

}  // namespace srf::internal::remote_descriptor
