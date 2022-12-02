/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <glog/logging.h>

#include <string>

namespace mrc::memory {

enum class memory_kind
{
    none,
    host,
    pinned,
    device,
    managed,
};

static const std::string& kind_string(const memory_kind& kind)
{
    switch (kind)
    {
    case memory_kind::none:
        static std::string none = "none";
        return none;
        break;

    case memory_kind::host:
        static std::string host = "host";
        return host;
        break;

    case memory_kind::pinned:
        static std::string pinned = "pinned";
        return pinned;
        break;

    case memory_kind::device:
        static std::string device = "device";
        return device;
        break;

    case memory_kind::managed:
        static std::string managed = "managed";
        return managed;
        break;
    }

    static std::string error = "error";
    LOG(FATAL) << "uknown memory_kind";
    return error;
}

}  // namespace mrc::memory
