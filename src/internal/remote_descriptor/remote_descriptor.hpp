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

#include "internal/remote_descriptor/encoded_object.hpp"
#include "internal/resources/forward.hpp"

#include "srf/codable/forward.hpp"
#include "srf/protos/codable.pb.h"
#include "srf/utils/macros.hpp"

#include <memory>
#include <stdexcept>

namespace srf::internal::remote_descriptor {

class Manager;

class RemoteDescriptor final
{
    RemoteDescriptor(std::shared_ptr<Manager> manager,
                     std::unique_ptr<srf::codable::protos::RemoteDescriptor> rd,
                     resources::PartitionResources& resources);

  public:
    RemoteDescriptor() = default;
    DELETE_COPYABILITY(RemoteDescriptor);
    DEFAULT_MOVEABILITY(RemoteDescriptor);

    ~RemoteDescriptor();

    operator bool() const;

    std::unique_ptr<const srf::codable::protos::RemoteDescriptor> release_ownership();

    void release();

    const EncodedObject& encoded_object() const;

    // template <typename T>
    // T decode()
    // {
    //     if (!m_descriptor)
    //     {
    //         throw std::runtime_error("unable decode empty descriptor");
    //     }

    //     codable::decode<T>(m_descriptor->encoded_object());
    // }

  private:
    std::unique_ptr<srf::codable::protos::RemoteDescriptor> m_descriptor;
    std::shared_ptr<Manager> m_manager;
    std::unique_ptr<EncodedObject> m_encoded_object;
    friend Manager;
};

}  // namespace srf::internal::remote_descriptor
