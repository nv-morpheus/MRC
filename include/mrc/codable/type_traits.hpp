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
#include "mrc/codable/encoding_options.hpp"
#include "mrc/utils/sfinae_concept.hpp"

#include <memory>
#include <type_traits>

namespace mrc::codable {

template <typename T>
class Encoder;

template <typename T>
class Decoder;

namespace detail {

template <typename T>
auto serialize(sfinae::full_concept c, const T& obj, Encoder<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize(obj, enc, opts), void);

template <typename T>
auto serialize(sfinae::l4_concept c, const T& obj, Encoder<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize(obj, enc), void);

template <typename T>
auto serialize(sfinae::l3_concept c, const T& obj, Encoder<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(enc.serialize(obj, opts), void);

template <typename T>
auto serialize(sfinae::l2_concept c, const T& obj, Encoder<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(enc.serialize(obj), void);

template <typename T>
void serialize(sfinae::error error, const T& obj, Encoder<T>& enc, const EncodingOptions& opts)
{
    static_assert(sfinae::invalid_concept<T>::error, "object is not encodable");
}

template <typename T>
auto deserialize(sfinae::full_concept c, const Decoder<T>& encoding, std::size_t object_idx)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::deserialize(encoding, object_idx), T);

template <typename T>
auto deserialize(sfinae::l4_concept c, const Decoder<T>& encoding, std::size_t object_idx)
    -> MRC_AUTO_RETURN_TYPE(T::deserialize(encoding, object_idx), T);

template <typename T>
sfinae::error deserialize(sfinae::error error, const Decoder<T>& encoding, std::size_t object_idx)
{
    static_assert(sfinae::invalid_concept<T>::error, "object is not decodable");
    return {};
}

}  // namespace detail

template <typename T, typename = void>
struct is_encodable : std::false_type
{};

template <typename T, typename = void>
struct is_decodable : std::false_type
{};

template <typename T>
struct is_encodable<T,
                    std::enable_if_t<std::is_same_v<
                        decltype(std::declval<codable_protocol<T>&>().serialize(
                            std::declval<T&>(), std::declval<Encoder<T>&>(), std::declval<const EncodingOptions&>())),
                        void>>> : std::true_type
{};

template <typename T>
struct is_encodable<T,
                    std::enable_if_t<std::is_same_v<decltype(std::declval<codable_protocol<T>&>().serialize(
                                                        std::declval<T&>(), std::declval<Encoder<T>&>())),
                                                    void>>> : std::true_type
{};

template <typename T>
struct is_encodable<
    T,
    std::enable_if_t<std::is_same_v<decltype(std::declval<T&>().serialize(std::declval<Encoder<T>&>(),
                                                                          std::declval<const EncodingOptions&>())),
                                    void>>> : std::true_type
{};

template <typename T>
struct is_encodable<
    T,
    std::enable_if_t<std::is_same_v<decltype(std::declval<T&>().serialize(std::declval<Encoder<T>&>())), void>>>
  : std::true_type
{};

template <typename T>
struct is_decodable<
    T,
    std::enable_if_t<std::is_same_v<decltype(std::declval<codable_protocol<T>&>().deserialize(
                                        std::declval<const Decoder<T>&>(), std::declval<std::size_t>())),
                                    T>>> : std::true_type
{};

template <typename T>
struct is_decodable<
    T,
    std::enable_if_t<
        std::is_same_v<decltype(T::deserialize(std::declval<const Decoder<T>&>(), std::declval<std::size_t>())), T>>>
  : std::true_type
{};

template <typename T>
struct is_codable
  : std::conditional<(is_encodable<T>::value && is_decodable<T>::value), std::true_type, std::false_type>::type
{};

template <typename T>
inline constexpr bool is_encodable_v = is_encodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_decodable_v = is_decodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_codable_v = is_codable<T>::value;  // NOLINT

}  // namespace mrc::codable
