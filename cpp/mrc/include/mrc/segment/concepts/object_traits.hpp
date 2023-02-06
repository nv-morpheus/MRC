/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//
// Created by drobison on 1/20/23.
//

#pragma once

#include "mrc/segment/object.hpp"
#include "mrc/type_traits.hpp"

#include <concepts>

namespace mrc {

template <typename T>
struct is_mrc_object_type : public std::false_type
{};

template <typename T>
struct is_mrc_object_type<mrc::segment::Object<T>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_mrc_object_v = is_mrc_object_type<T>::value;  // NOLINT

template <typename T>
struct is_mrc_object_shared_pointer : public std::false_type
{};

template <typename T>
struct is_mrc_object_shared_pointer<std::shared_ptr<mrc::segment::Object<T>>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_mrc_object_shared_ptr_v = is_mrc_object_shared_pointer<T>::value;  // NOLINT

struct mrc_object_null_type
{
    using source_type_t = void;
    using sink_type_t   = void;
};

template <typename T>
struct mrc_object_sptr_type
{
    using type_t = mrc_object_null_type;
};

template <typename T>
struct mrc_object_sptr_type<std::shared_ptr<mrc::segment::Object<T>>>
{
    using type_t = T;
};

template <typename T>
using mrc_object_sptr_type_t = typename mrc_object_sptr_type<T>::type_t;

template <typename TypeT>
concept MRCObject = is_mrc_object_v<TypeT>;

template <typename TypeT>
concept MRCObjectSharedPtr = is_mrc_object_shared_ptr_v<TypeT>;

template <typename TypeT>
concept MRCObjProp = std::is_same_v<std::decay_t<TypeT>, mrc::segment::ObjectProperties>;

template <typename TypeT>
concept MRCObjPropSharedPtr = std::is_same_v<std::decay_t<TypeT>, std::shared_ptr<mrc::segment::ObjectProperties>>;

template <typename TypeT>
concept MRCObjectProxy = MRCObject<TypeT> || MRCObjectSharedPtr<TypeT> || MRCObjProp<TypeT> ||
                         MRCObjPropSharedPtr<TypeT> || std::is_convertible_v<TypeT, std::string>;

}  // namespace mrc