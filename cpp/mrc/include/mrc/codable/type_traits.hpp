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
#include "mrc/codable/encoding_options.hpp"
#include "mrc/utils/sfinae_concept.hpp"

#include <concepts>
#include <memory>
#include <type_traits>

namespace mrc::codable {

template <typename T>
class Encoder;

template <typename T>
class Decoder;

template <typename T>
class Encoder2;

template <typename T>
class Decoder2;

namespace detail {

template <typename T>
auto serialize(sfinae::full_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize(obj, enc, opts), void);

template <typename T>
auto serialize(sfinae::l4_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize(obj, enc), void);

template <typename T>
auto serialize(sfinae::l3_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(enc.serialize(obj, opts), void);

template <typename T>
auto serialize(sfinae::l2_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
    -> MRC_AUTO_RETURN_TYPE(enc.serialize(obj), void);

template <typename T>
void serialize(sfinae::error error, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
{
    static_assert(sfinae::invalid_concept<T>::error, "object is not encodable");
}

// template <typename T>
// auto serialize2(sfinae::full_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
//     -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize2(obj, enc, opts), void);

// template <typename T>
// auto serialize2(sfinae::l4_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
//     -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::serialize2(obj, enc), void);

// template <typename T>
// auto serialize2(sfinae::l3_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
//     -> MRC_AUTO_RETURN_TYPE(obj.serialize2(opts), void);

// template <typename T>
// auto serialize2(sfinae::l2_concept c, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
//     -> MRC_AUTO_RETURN_TYPE(obj.serialize2(), void);

// template <typename T>
// void serialize2(sfinae::error error, const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
// {
//     static_assert(sfinae::invalid_concept<T>::error, "object is not encodable");
// }

template <typename T>
auto deserialize(sfinae::full_concept c, const Decoder2<T>& encoding)
    -> MRC_AUTO_RETURN_TYPE(codable_protocol<T>::deserialize(encoding), T);

template <typename T>
auto deserialize(sfinae::l4_concept c, const Decoder2<T>& encoding)
    -> MRC_AUTO_RETURN_TYPE(T::deserialize(encoding), T);

template <typename T>
sfinae::error deserialize(sfinae::error error, const Decoder2<T>& encoding)
{
    static_assert(sfinae::invalid_concept<T>::error, "object is not decodable");
    return {};
}

}  // namespace detail

template <typename T, typename = void>
struct is_protocol_encodable : std::false_type
{};

template <typename T, typename = void>
struct is_member_encodable : std::false_type
{};

template <typename T, typename = void>
struct is_protocol_decodable : std::false_type
{};

template <typename T, typename = void>
struct is_static_decodable : std::false_type
{};

template <typename T>
struct is_protocol_encodable<T,
                             std::enable_if_t<std::is_same_v<decltype(std::declval<codable_protocol<T>&>().serialize(
                                                                 std::declval<T&>(),
                                                                 std::declval<Encoder2<T>&>(),
                                                                 std::declval<const EncodingOptions&>())),
                                                             void>>> : std::true_type
{};

template <typename T>
struct is_protocol_encodable<
    T,
    std::enable_if_t<std::is_same_v<
        decltype(std::declval<codable_protocol<T>&>().serialize(std::declval<T&>(), std::declval<Encoder2<T>&>())),
        void>>> : std::true_type
{};

template <typename T>
struct is_member_encodable<
    T,
    std::enable_if_t<std::is_same_v<decltype(std::declval<const T&>().serialize(std::declval<Encoder2<T>&>(),
                                                                                std::declval<const EncodingOptions&>())),
                                    void>>> : std::true_type
{};

template <typename T>
struct is_member_encodable<
    T,
    std::enable_if_t<std::is_same_v<decltype(std::declval<const T&>().serialize(std::declval<Encoder2<T>&>())), void>>>
  : std::true_type
{};

template <typename T>
inline constexpr bool is_protocol_encodable_v = is_protocol_encodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_member_encodable_v = is_member_encodable<T>::value;  // NOLINT

template <typename T>
struct is_protocol_decodable<T,
                             std::enable_if_t<std::is_same_v<decltype(std::declval<codable_protocol<T>&>().deserialize(
                                                                 std::declval<const Decoder2<T>&>())),
                                                             T>>> : std::true_type
{};

template <typename T>
struct is_static_decodable<
    T,
    std::enable_if_t<
        std::is_same_v<decltype(T::deserialize(std::declval<const Decoder2<T>&>())), T>>>
  : std::true_type
{};

template <typename T>
inline constexpr bool is_protocol_decodable_v = is_protocol_decodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_static_decodable_v = is_static_decodable<T>::value;  // NOLINT

// template <typename T>
// concept is_static_decodable = requires(T) {
//     {
//         T::deserialize(std::declval<Decoder<T>&>(), std::declval<std::size_t>())
//     } -> std::same_as<T>;
// };

template <typename T>
struct is_encodable
  : std::conditional_t<(is_protocol_encodable_v<T> || is_member_encodable_v<T>), std::true_type, std::false_type>
{};

template <typename T>
struct is_decodable
  : std::conditional_t<(is_protocol_decodable_v<T> || is_static_decodable_v<T>), std::true_type, std::false_type>
{};

template <typename T>
struct is_codable
  : std::conditional_t<(is_encodable<T>::value && is_decodable<T>::value), std::true_type, std::false_type>
{};

template <typename T>
inline constexpr bool is_encodable_v = is_encodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_decodable_v = is_decodable<T>::value;  // NOLINT

template <typename T>
inline constexpr bool is_codable_v = is_codable<T>::value;  // NOLINT

template <typename T>
concept protocol_encodable = is_protocol_encodable_v<T>;

template <typename T>
concept member_encodable = is_member_encodable_v<T>;

template <typename T>
concept protocol_decodable = is_protocol_decodable_v<T>;

template <typename T>
concept static_decodable = is_static_decodable_v<T>;

template <typename T>
concept encodable = is_encodable_v<T>;

template <typename T>
concept decodable = is_decodable_v<T>;

template <typename T>
concept codable = is_codable_v<T>;

namespace detail {
template <protocol_encodable T>
auto serialize2(const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
{
    return codable_protocol<T>::serialize(obj, enc, opts);
};

template <member_encodable T>
auto serialize2(const T& obj, Encoder2<T>& enc, const EncodingOptions& opts)
{
    return obj.serialize(enc, opts);
};

template <protocol_decodable T>
auto deserialize2(const Decoder2<T>& decoder)
{
    return codable_protocol<T>::deserialize(decoder);
};

template <static_decodable T>
auto deserialize2(const Decoder2<T>& decoder)
{
    return T::deserialize(decoder);
};

}  // namespace detail

}  // namespace mrc::codable
