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

#include "srf/codable/forward.hpp"

#include <memory>

namespace srf::internal::remote_descriptor {

class Manager;

class RemoteDescriptor final
{
    RemoteDescriptor(std::shared_ptr<Manager> manager, std::unique_ptr<srf::codable::protos::RemoteDescriptor> rd);

  public:
    ~RemoteDescriptor();

    operator bool() const
    {
        return bool(m_descriptor);
    }

    void release();

  private:
    std::unique_ptr<srf::codable::protos::RemoteDescriptor> m_descriptor;
    std::shared_ptr<Manager> m_manager;
    friend Manager;
};

}  // namespace srf::internal::remote_descriptor
