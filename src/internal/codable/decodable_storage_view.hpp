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

#include "internal/codable/storage_resources.hpp"
#include "internal/codable/storage_view.hpp"
#include "internal/resources/forward.hpp"
#include "internal/ucx/memory_block.hpp"
#include "internal/utils/contains.hpp"

#include "srf/codable/api.hpp"
#include "srf/memory/buffer.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"
#include "srf/memory/resources/memory_resource.hpp"
#include "srf/protos/codable.pb.h"

#include <glog/logging.h>

#include <mutex>
#include <optional>
#include <typeindex>

namespace srf::internal::codable {

/**
 * @brief Storage implements the IDecodableStorage interface for an EncodedObject/Storage
 */
class DecodableStorageView : public virtual srf::codable::IDecodableStorage, public IStorageResources
{
  public:
    ~DecodableStorageView() override = default;

  protected:
    void copy_from_buffer(const idx_t& idx, srf::memory::buffer_view dst_view) const final;

    std::size_t buffer_size(const idx_t& idx) const final;

    void copy_from_registered_buffer(const idx_t& idx, srf::memory::buffer_view& dst_view) const;

    void copy_from_eager_buffer(const idx_t& idx, srf::memory::buffer_view& dst_view) const;

    std::shared_ptr<srf::memory::memory_resource> host_memory_resource() const final;

    std::shared_ptr<srf::memory::memory_resource> device_memory_resource() const final;
};

}  // namespace srf::internal::codable
