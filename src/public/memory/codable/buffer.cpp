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

#include "mrc/memory/codable/buffer.hpp"

#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/encoding_options.hpp"

#include <glog/logging.h>

#include <optional>
#include <typeindex>
#include <utility>

namespace mrc::codable {

void codable_protocol<mrc::memory::buffer>::serialize(const memory::buffer& obj,
                                                      Encoder<memory::buffer>& encoded,
                                                      const EncodingOptions& opts)
{
    auto idx = encoded.register_memory_view(obj);
    if (!idx)
    {
        encoded.copy_to_eager_descriptor(obj);
    }
}

memory::buffer codable_protocol<mrc::memory::buffer>::deserialize(const Decoder<memory::buffer>& encoded,
                                                                  std::size_t object_idx)
{
    DCHECK_EQ(std::type_index(typeid(memory::buffer)).hash_code(), encoded.type_index_hash_for_object(object_idx));
    auto idx   = encoded.start_idx_for_object(object_idx);
    auto bytes = encoded.buffer_size(idx);

    mrc::memory::buffer buffer(bytes, encoded.host_memory_resource());
    encoded.copy_from_buffer(idx, buffer);

    return std::move(buffer);
}

}  // namespace mrc::codable
