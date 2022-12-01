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

// #include "internal/remote_descriptor/handle.hpp"
// #include "internal/resources/forward.hpp"

// #include "mrc/codable/api.hpp"
// #include "mrc/protos/codable.pb.h"
// #include "mrc/runtime/remote_descriptor.hpp"
// #include "mrc/types.hpp"
// #include "mrc/utils/macros.hpp"

// #include <memory>
// #include <stdexcept>
// #include <utility>

namespace mrc::internal::remote_descriptor {

// class Manager;

// class RemoteDescriptor final
// {
//     RemoteDescriptor(std::shared_ptr<Manager> manager, Handle&& handle) :
//       m_manager(std::move(manager)),
//       m_handle(std::move(handle))
//     {}

//   public:
//     static std::unique_ptr<RemoteDescriptor> unwrap(mrc::remote_descriptor::RemoteDescriptor&& rd);

//     ~RemoteDescriptor();

//     DELETE_COPYABILITY(RemoteDescriptor);
//     DEFAULT_MOVEABILITY(RemoteDescriptor);

//     /**
//      * @brief Returns false if the RemoteDescriptor was released or ownership was transfered; otherwise, returns
//      true.
//      */
//     operator bool() const;

//     /**
//      * @brief Transfer ownership of the RemoteDescriptor and its global tokens to a RemoteDescriptorHandle
//      *
//      * @return RemoteDescriptorHandle
//      */
//     Handle transfer_ownership();

//     /**
//      * @brief Release ownership of the RemoteDescriptor and decrement the global tokens by the number of tokens held
//      by
//      * this descriptor.
//      */
//     void release_ownership();

//     /**
//      * @brief Returns a reference to an IDecodableStorage for decoding a RemoteDescriptor
//      */
//     const mrc::codable::IDecodableStorage& encoding() const;

//   private:
//     std::shared_ptr<Manager> m_manager;
//     std::shared_ptr<mrc::codable::IDecodableStorage> m_encoding;
//     InstanceID m_instance_id;

//     friend Manager;
// };

}  // namespace mrc::internal::remote_descriptor
