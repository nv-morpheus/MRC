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

#include "srf/codable/encoded_object.hpp"
#include "srf/codable/type_traits.hpp"
#include "srf/utils/sfinae_concept.hpp"

#include <memory>

namespace srf::codable {

template <typename T>
struct Decoder
{
    static T deserialize(const EncodedObject& encoding, std::size_t object_idx)
    {
        return detail::deserialize<T>(sfinae::full_concept{}, encoding, object_idx);
    }
};

template <typename T>
auto decode(const EncodedObject& encoding, std::size_t object_idx = 0)
{
    return Decoder<T>::deserialize(encoding, object_idx);
}

}  // namespace srf::codable
