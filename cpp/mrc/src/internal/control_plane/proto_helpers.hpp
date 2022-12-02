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

#include "internal/expected.hpp"

#include <google/protobuf/any.pb.h>

#include <set>

namespace mrc::internal::control_plane {

// protobuf convenience methods
template <typename T>
Expected<std::set<T>> check_unique_repeated_field(const google::protobuf::RepeatedPtrField<T>& items)
{
    std::set<T> unique(items.begin(), items.end());
    if (unique.size() != items.size())
    {
        return Error::create("non-unique repeated field; duplicated detected");
    }
    return unique;
}

template <typename T>
Expected<T> unpack(const google::protobuf::Any& message)
{
    T msg;
    if (message.UnpackTo(&msg))
    {
        return msg;
    }
    return Error::create("unable to unpack message to the requested type");
}

}  // namespace mrc::internal::control_plane
