/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/codable/types.hpp"
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

    static void serialize(const T& t, Encoder2<T>& encoder, const EncodingOptions& opts)
    {
        // codable::encode2(t, encoder, std::move(opts));

        encoder.write_descriptor({&t, sizeof(t), memory::memory_kind::host}, DescriptorKind::Eager);
    }

    static T deserialize(const Decoder<T>& decoder, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(T)).hash_code(), decoder.type_index_hash_for_object(object_idx));
        auto idx = decoder.start_idx_for_object(object_idx);

        T val;
        decoder.copy_from_buffer(object_idx, {&val, sizeof(T), memory::memory_kind::host});

        return val;
    }

    static T deserialize(const Decoder2<T>& decoder, std::size_t object_idx)
    {
        // DCHECK_EQ(std::type_index(typeid(T)).hash_code(), decoder.type_index_hash_for_object(object_idx));
        // auto idx = decoder.start_idx_for_object(object_idx);

        T val;
        decoder.read_descriptor(0, {&val, sizeof(T), memory::memory_kind::host});

        return val;
    }
};

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_same_v<T, std::string>>>
{
    static void serialize(const std::string& str, Encoder<T>& encoder, const EncodingOptions& opts)
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

    static void serialize(const std::string& str, Encoder2<T>& encoder, const EncodingOptions& opts)
    {
        DescriptorKind kind = DescriptorKind::Default;

        encoder.write_descriptor({str.data(), str.size(), memory::memory_kind::host}, kind);

        // if (opts.force_copy())
        // {
        //     auto index = encoder.create_memory_buffer(str.size());
        //     encoder.copy_to_buffer(index, {str.data(), str.size(), memory::memory_kind::host});
        // }
        // else
        // {
        //     auto idx = encoder.register_memory_view({str.data(), str.size(), memory::memory_kind::host});
        //     if (!idx)
        //     {
        //         encoder.copy_to_eager_descriptor({str.data(), str.size(), memory::memory_kind::host});
        //     }
        // }
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

    static T deserialize(const Decoder2<T>& decoder, std::size_t object_idx)
    {
        // DCHECK_EQ(std::type_index(typeid(T)).hash_code(), decoder.type_index_hash_for_object(object_idx));
        // auto idx   = decoder.start_idx_for_object(object_idx);
        auto bytes = decoder.descriptor_size(0);

        T str;
        str.resize(bytes);
        decoder.read_descriptor(0, {str.data(), str.size(), memory::memory_kind::host});

        return str;
    }
};

template <typename T>
struct codable_protocol<std::vector<T>>
{
    static void serialize(const std::vector<T>& obj,
                          mrc::codable::Encoder<std::vector<T>>& encoder,
                          const mrc::codable::EncodingOptions& opts)
    {
        // First put in the size
        mrc::codable::encode2(obj.size(), encoder, opts);

        // Now encode each object
        for (const auto& o : obj)
        {
            mrc::codable::encode2(o, encoder, opts);
        }
    }

    static void serialize(const std::vector<T>& obj,
                          mrc::codable::Encoder2<std::vector<T>>& encoder,
                          const mrc::codable::EncodingOptions& opts)
    {
        // First put in the size
        mrc::codable::encode2(obj.size(), encoder, opts);

        if constexpr (std::is_fundamental_v<T>)
        {
            // Since these are fundamental types, just encode in a single memory block
            encoder.write_descriptor({obj.data(), obj.size() * sizeof(T), memory::memory_kind::host},
                                     DescriptorKind::Deferred);
        }
        else
        {
            // Now encode each object
            for (const auto& o : obj)
            {
                mrc::codable::encode2(o, encoder, opts);
            }
        }
    }

    static std::vector<T> deserialize(const Decoder<std::vector<T>>& decoder, std::size_t object_idx)
    {
        DCHECK_EQ(std::type_index(typeid(std::vector<T>)).hash_code(), decoder.type_index_hash_for_object(object_idx));

        auto count = mrc::codable::decode2<size_t>(decoder, object_idx);

        auto object = std::vector<T>(count);

        auto idx   = decoder.start_idx_for_object(object_idx);
        auto bytes = decoder.buffer_size(idx);

        decoder.copy_from_buffer(idx, {object.data(), count * sizeof(T), memory::memory_kind::host});

        return object;
    }

    static std::vector<T> deserialize(const Decoder2<std::vector<T>>& decoder, std::size_t object_idx)
    {
        // DCHECK_EQ(std::type_index(typeid(std::vector<T>)).hash_code(),
        // decoder.type_index_hash_for_object(object_idx));

        auto count = mrc::codable::decode2<size_t>(decoder, object_idx);

        auto object = std::vector<T>(count);

        decoder.read_descriptor(0, {object.data(), count * sizeof(T), memory::memory_kind::host});

        return object;
    }
};

}  // namespace mrc::codable
