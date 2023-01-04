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

namespace mrc::codable {

class StorageForwarder : public virtual IStorage
{
  public:
    ~StorageForwarder() override = default;

    /**
     * @brief ObjectDescriptor describing the encoded object.
     * @return const protos::ObjectDescriptor&
     */
    const protos::EncodedObject& proto() const final
    {
        return const_storage().proto();
    }

    /**
     * @brief The number of unqiue objects described by the encoded object
     * @return std::size_t
     */
    obj_idx_t object_count() const final
    {
        return const_storage().object_count();
    }

    /**
     * @brief The number of unique memory regions contained in the multiple part descriptor.
     * @return std::size_t
     */
    idx_t descriptor_count() const final
    {
        return const_storage().object_count();
    }

    /**
     * @brief Hash of std::type_index for the object at idx
     *
     * @param object_idx
     * @return std::type_index
     */
    std::size_t type_index_hash_for_object(const obj_idx_t& object_idx) const final
    {
        return const_storage().type_index_hash_for_object(object_idx);
    }

    /**
     * @brief Starting index of object at idx
     *
     * @param object_idx
     * @return idx_t
     */
    idx_t start_idx_for_object(const obj_idx_t& object_idx) const final
    {
        return const_storage().start_idx_for_object(object_idx);
    }

    /**
     * @brief Parent for object at idx
     *
     * @return std::optional<obj_idx_t> - if nullopt, then the object is a top-level object; otherwise, it is a child
     * object with a parent object at the returned value
     */
    std::optional<obj_idx_t> parent_obj_idx_for_object(const obj_idx_t& object_idx) const final
    {
        return const_storage().parent_obj_idx_for_object(object_idx);
    }

  private:
    virtual const IStorage& const_storage() const = 0;
};

}  // namespace mrc::codable
