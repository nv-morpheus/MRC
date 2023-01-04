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

#include "mrc/codable/codable_protocol.hpp"
#include "mrc/codable/decode.hpp"
#include "mrc/codable/encode.hpp"
#include "mrc/codable/encoding_options.hpp"
#include "mrc/memory/buffer_view.hpp"
#include "mrc/memory/memory_kind.hpp"

#include <type_traits>
#include <typeindex>

namespace mrc::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_fundamental_v<T>>>
{
    static void serialize(const T& t, Encoder<T>& encoder, const EncodingOptions& opts)
    {
        auto index = encoder.copy_to_eager_descriptor({&t, sizeof(t), memory::memory_kind::host});
    }

    static T deserialize(const Decoder<T>& decoder, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), decoder.type_index_hash_for_object(object_idx));
        auto idx = decoder.start_idx_for_object(object_idx);

        T val;
        decoder.copy_from_buffer(object_idx, {&val, sizeof(T), memory::memory_kind::host});

        return val;
    }
};

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, std::string>>>
{
    static void serialize(const T& str, Encoder<T>& encoder, const EncodingOptions& opts)
    {
        if (opts.force_copy())
        {
            auto index = encoder.create_memory_buffer(str.size());
            encoder.copy_to_buffer(index, {str.data(), str.size(), memory::memory_kind::host});
        }
        else
        {
            auto idx = encoder.register_memory_view({str.data(), str.size(), memory::memory_kind::host});
            if (!idx)
            {
                encoder.copy_to_eager_descriptor({str.data(), str.size(), memory::memory_kind::host});
            }
        }
    }

    static T deserialize(const Decoder<T>& decoder, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), decoder.type_index_hash_for_object(object_idx));
        auto idx   = decoder.start_idx_for_object(object_idx);
        auto bytes = decoder.buffer_size(idx);

        T str;
        str.resize(bytes);
        decoder.copy_from_buffer(idx, {str.data(), str.size(), memory::memory_kind::host});

        return str;
    }
};

}  // namespace mrc::codable
