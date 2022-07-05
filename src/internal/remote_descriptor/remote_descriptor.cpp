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

#include "internal/remote_descriptor/remote_descriptor.hpp"

#include "internal/remote_descriptor/manager.hpp"

namespace srf::internal::remote_descriptor {

RemoteDescriptor::~RemoteDescriptor()
{
    release();
}

void RemoteDescriptor::release()
{
    if (m_descriptor)
    {
        CHECK(m_manager);
        m_manager->decrement_tokens(m_descriptor->object_id(), m_descriptor->tokens());
        m_manager.reset();
        m_descriptor.reset();
    }
}
RemoteDescriptor::RemoteDescriptor(std::shared_ptr<Manager> manager,
                                   std::unique_ptr<srf::codable::protos::RemoteDescriptor> rd) :
  m_manager(std::move(manager)),
  m_descriptor(std::move(rd))
{}
}  // namespace srf::internal::remote_descriptor
