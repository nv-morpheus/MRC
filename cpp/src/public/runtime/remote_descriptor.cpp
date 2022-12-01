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

#include "mrc/runtime/remote_descriptor.hpp"

#include "mrc/runtime/remote_descriptor_handle.hpp"
#include "mrc/runtime/remote_descriptor_manager.hpp"

#include <utility>

namespace mrc::runtime {

RemoteDescriptor::RemoteDescriptor(std::shared_ptr<IRemoteDescriptorManager> manager,
                                   std::unique_ptr<IRemoteDescriptorHandle> handle) :
  m_manager(std::move(manager)),
  m_handle(std::move(handle))
{}

RemoteDescriptor::~RemoteDescriptor() = default;

bool RemoteDescriptor::has_value() const
{
    return (m_manager && m_handle);
}

void RemoteDescriptor::release_ownership()
{
    if (m_manager)
    {
        CHECK(m_handle);
        m_manager->release_handle(std::move(m_handle));
        m_manager.reset();
    }
}

std::unique_ptr<IRemoteDescriptorHandle> RemoteDescriptor::release_handle()
{
    CHECK(*this);
    m_manager.reset();
    return std::move(m_handle);
}

}  // namespace mrc::runtime
