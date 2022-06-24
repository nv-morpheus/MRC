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

#include <google/protobuf/message.h>

#include <type_traits>

namespace srf::codable {

template <typename T>
struct codable_protocol<T, std::enable_if_t<std::is_base_of_v<::google::protobuf::Message, T>>>
{
    static void serialize(const T& msg, Encoded<T>& encoded, const EncodingOptions& opts)
    {
        auto guard = encoded.acquire_encoding_context();
        auto index = encoded.add_host_buffer(msg.ByteSizeLong());
        auto block = encoded.mutable_memory_block(index);
        msg.SerializeToArray(block.data(), block.bytes());
    }

    static T deserialize(const EncodedObject& encoded, std::size_t object_idx)
    {
        T msg;
        auto idx          = encoded.start_idx_for_object(object_idx);
        const auto& block = encoded.memory_block(idx);
        CHECK(msg.ParseFromArray(block.data(), block.bytes()));
        return msg;
    }
};

}  // namespace srf::codable
