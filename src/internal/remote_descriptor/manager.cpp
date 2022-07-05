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

#include "internal/remote_descriptor/manager.hpp"

#include "internal/remote_descriptor/remote_descriptor.hpp"
#include "internal/remote_descriptor/storage.hpp"

#include "srf/protos/codable.pb.h"

namespace srf::internal::remote_descriptor {

void Manager::decrement_tokens(std::size_t object_id, std::size_t token_count)
{
    DVLOG(10) << "decrementing " << token_count << " tokens from object_id: " << object_id;
    auto search = m_stored_objects.find(object_id);
    CHECK(search != m_stored_objects.end());
    auto remaining = search->second->decrement_tokens(token_count);
    if (remaining == 0)
    {
        DVLOG(10) << "destroying object_id: " << object_id;
        m_stored_objects.erase(search);
    }
}

RemoteDescriptor Manager::store_object(std::unique_ptr<Storage> object)
{
    CHECK(object);

    auto object_id = reinterpret_cast<std::size_t>(object.get());
    auto rd        = std::make_unique<srf::codable::protos::RemoteDescriptor>();

    DVLOG(10) << "storing object_id: " << object_id << " with " << object->tokens_count() << " tokens";

    // rd->set_instance_id(/* todo */);
    rd->set_object_id(object_id);
    rd->set_tokens(object->tokens_count());
    *(rd->mutable_encoded_object()) = object->encoded_object().proto();
    m_stored_objects[object_id]     = std::move(object);

    return RemoteDescriptor(shared_from_this(), std::move(rd));
}

std::size_t Manager::size() const
{
    return m_stored_objects.size();
}
}  // namespace srf::internal::remote_descriptor
