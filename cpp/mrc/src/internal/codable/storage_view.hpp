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

#include "mrc/codable/api.hpp"
#include "mrc/protos/codable.pb.h"  // todo(iwyu) protos::EncodedObject should be forward declared

#include <cstddef>
#include <optional>

namespace mrc::internal::codable {

/**
 * @brief StorageView implements the methods on IEncodedObject with a reference to a protos::EncodedObject as the
 * backing storage.
 *
 * This object does not own the protos::EncodedObject.
 *
 * This is mostly used to avoid the repeatative implementation of the interface when using a unique_ptr, shared_ptr or a
 * direct protos::EncodedObject.
 */
class StorageView : public virtual mrc::codable::IStorage
{
  public:
    StorageView()           = default;
    ~StorageView() override = default;

    const mrc::codable::protos::EncodedObject& proto() const final;

    obj_idx_t object_count() const final;

    idx_t descriptor_count() const final;

    std::size_t type_index_hash_for_object(const obj_idx_t& object_idx) const final;

    idx_t start_idx_for_object(const obj_idx_t& object_idx) const final;

    std::optional<obj_idx_t> parent_obj_idx_for_object(const obj_idx_t& object_idx) const final;

  private:
    virtual const mrc::codable::protos::EncodedObject& get_proto() const = 0;
};

}  // namespace mrc::internal::codable
