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
#include "srf/memory/block.hpp"
#include "srf/memory/memory_kind.hpp"

#include <type_traits>
#include <typeindex>

namespace srf::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_fundamental_v<T>>>
{
    static void serialize(const T& t, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        auto guard = encoded.acquire_encoding_context();
        auto index = encoded.add_eager_buffer(&t, sizeof(t));
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));
        auto idx          = encoded.start_idx_for_object(object_idx);
        const auto& eager = encoded.eager_descriptor(idx);
        T val             = *(reinterpret_cast<const T*>(eager.data().data()));
        return val;
    }
};

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, std::string>>>
{
    static void serialize(const T& str, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        auto guard = encoded.acquire_encoding_context();
        if (opts.force_copy())
        {
            auto index = encoded.add_host_buffer(str.size());
            auto block = encoded.mutable_memory_block(index);
            std::memcpy(block.data(), str.data(), str.size());
        }
        else
        {
            // not registered
            encoded.add_memory_block(memory::const_block(str.data(), str.size(), memory::memory_kind_type::host));
        }
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), encoded.type_index_hash_for_object(object_idx));
        T str;
        auto idx           = encoded.start_idx_for_object(object_idx);
        const auto& buffer = encoded.memory_block(idx);

        str.resize(buffer.bytes());
        std::memcpy(str.data(), buffer.data(), buffer.bytes());

        return str;
    }
};

}  // namespace srf::codable
