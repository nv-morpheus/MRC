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

#include <srf/codable/encoded_object.hpp>
#include <srf/codable/type_traits.hpp>
#include <srf/utils/sfinae_concept.hpp>

#include <memory>

namespace srf::codable {

template <typename T>
struct Encoder
{
    static void serialize(const T& t, Encoded<T>& enc, const EncodingOptions& opts = {})
    {
        return detail::serialize(sfinae::full_concept{}, t, enc, opts);
    }
};

template <typename T>
auto encode(const T& t, EncodingOptions opts = {})
{
    auto encoded = std::make_unique<Encoded<T>>();
    Encoder<T>::serialize(t, *encoded, std::move(opts));
    return std::move(encoded);
}

template <typename T>
void encode(const T& t, EncodedObject& encoding, EncodingOptions opts = {})
{
    auto enc = reinterpret_cast<Encoded<T>*>(&encoding);
    Encoder<T>::serialize(t, *enc, std::move(opts));
}

template <typename T>
void encode(const T& t, Encoded<T>& enc, EncodingOptions opts = {})
{
    Encoder<T>::serialize(t, enc, std::move(opts));
}

template <typename T>
void encode(const T& t, Encoded<T>* enc, EncodingOptions opts = {})
{
    CHECK(enc);
    Encoder<T>::serialize(t, *enc, std::move(opts));
}

}  // namespace srf::codable
