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

// #include "internal/remote_descriptor/manager.hpp"
// #include "internal/resources/partition_resources.hpp"

// #include "mrc/codable/api.hpp"
// #include "mrc/protos/codable.pb.h"

namespace mrc::internal::remote_descriptor {

// RemoteDescriptor::~RemoteDescriptor()
// {
//     release_ownership();
// }

// void RemoteDescriptor::release_ownership()
// {
//     if (m_manager)
//     {
//         m_manager->decrement_tokens(std::move(m_handle));
//         m_manager.reset();
//     }
// }

// RemoteDescriptor::operator bool() const
// {
//     return bool(m_manager);
// }

// Handle RemoteDescriptor::transfer_ownership()
// {
//     CHECK(*this);
//     m_manager.reset();
//     return std::move(m_handle);
// }

// // const mrc::codable::IDecodableStorage& RemoteDescriptor::encoding() const
// // {
// //     CHECK(*this);
// //     return m_handle.proto().encoded_object();
// // }

// std::unique_ptr<RemoteDescriptor> RemoteDescriptor::unwrap(mrc::runtime::RemoteDescriptor&& rd)
// {
//     return std::move(rd.m_impl);
// }

}  // namespace mrc::internal::remote_descriptor
