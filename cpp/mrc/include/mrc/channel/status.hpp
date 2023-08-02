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

#include <ostream>

namespace mrc::channel {

enum class Status
{
    success = 0,
    empty,
    full,
    closed,
    timeout,
    error
};

static inline std::ostream& operator<<(std::ostream& os, const Status& s)
{
    switch (s)
    {
    case Status::success:
        return os << "success";
    case Status::empty:
        return os << "empty";
    case Status::full:
        return os << "full";
    case Status::closed:
        return os << "closed";
    case Status::timeout:
        return os << "timeout";
    case Status::error:
        return os << "error";
    default:
        throw std::logic_error("Unsupported channel::Status enum. Was a new value added recently?");
    }
}

}  // namespace mrc::channel
