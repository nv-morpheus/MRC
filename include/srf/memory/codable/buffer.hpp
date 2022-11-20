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

#include "srf/codable/codable_protocol.hpp"
#include "srf/codable/encoded_object.hpp"
#include "srf/codable/encoding_options.hpp"
#include "srf/memory/buffer_view.hpp"
#include "srf/memory/memory_kind.hpp"

#include <type_traits>
#include <typeindex>

namespace srf::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, srf::memory::buffer>>>
{
    static void serialize(const T& obj, Encoder<T>& encoded, const EncodingOptions& opts)
    {
        auto idx = encoded.register_memory_view(obj);
        if (!idx)
        {
            encoded.copy_to_eager_descriptor(obj);
        }
    }

    static T deserialize(const Decoder<T>& encoded, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));
        auto idx   = encoded.start_idx_for_object(object_idx);
        auto bytes = encoded.buffer_size(idx);

        srf::memory::buffer buffer(bytes, encoded.host_memory_resource());
        encoded.copy_from_buffer(idx, buffer);

        return buffer;
    }
};

}  // namespace srf::codable
