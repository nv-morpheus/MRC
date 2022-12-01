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

#include "internal/codable/storage_view.hpp"

#include <glog/logging.h>

#include <optional>

namespace mrc::internal::codable {

const mrc::codable::protos::EncodedObject& StorageView::proto() const
{
    return get_proto();
}

StorageView::obj_idx_t StorageView::object_count() const
{
    return get_proto().objects_size();
}

StorageView::idx_t StorageView::descriptor_count() const
{
    return get_proto().descriptors_size();
}

std::size_t StorageView::type_index_hash_for_object(const obj_idx_t& object_idx) const
{
    CHECK_LT(object_idx, get_proto().objects_size());
    return get_proto().objects().at(object_idx).type_index_hash();
}

StorageView::idx_t StorageView::start_idx_for_object(const obj_idx_t& object_idx) const
{
    CHECK_LT(object_idx, get_proto().objects_size());
    return get_proto().objects().at(object_idx).starting_descriptor_idx();
}

std::optional<StorageView::obj_idx_t> StorageView::parent_obj_idx_for_object(const obj_idx_t& object_idx) const
{
    CHECK_LT(object_idx, get_proto().objects_size());
    auto parent_object_idx = get_proto().objects().at(object_idx).parent_object_idx();
    if (parent_object_idx < 0)
    {
        return std::nullopt;
    }
    return parent_object_idx;
}

}  // namespace mrc::internal::codable
