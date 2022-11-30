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

#include <google/protobuf/message.h>

#include <type_traits>

namespace mrc::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_base_of_v<::google::protobuf::Message, T>>>
{
    static void serialize(const T& msg, Encoder<T>& encoder, const EncodingOptions& opts)
    {
        auto index = encoder.add_host_buffer(msg.ByteSizeLong());
        auto block = encoder.mutable_memory_block(index);
        msg.SerializeToArray(block.data(), block.bytes());
    }

    static T deserialize(const Decoder<T>& decoder, std::size_t object_idx)
    {
        T msg;
        auto idx          = decoder.start_idx_for_object(object_idx);
        const auto& block = decoder.memory_block(idx);
        CHECK(msg.ParseFromArray(block.data(), block.bytes()));
        return msg;
    }
};

}  // namespace mrc::codable
