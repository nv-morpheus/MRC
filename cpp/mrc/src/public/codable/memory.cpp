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

#include "mrc/memory/memory_kind.hpp"
#include "mrc/protos/codable.pb.h"

#include <glog/logging.h>

#include <ostream>

namespace mrc::codable {

memory::memory_kind decode_memory_type(const protos::MemoryKind& proto_kind)
{
    switch (proto_kind)
    {
    case protos::MemoryKind::Host:
        return memory::memory_kind::host;
    case protos::MemoryKind::Pinned:
        return memory::memory_kind::pinned;
    case protos::MemoryKind::Device:
        return memory::memory_kind::device;
    case protos::MemoryKind::Managed:
        return memory::memory_kind::managed;
    default:
        LOG(FATAL) << "unhandled protos::MemoryKind";
    };

    return memory::memory_kind::none;
}

protos::MemoryKind encode_memory_type(memory::memory_kind mem_kind)
{
    switch (mem_kind)
    {
    case memory::memory_kind::host:
        return protos::MemoryKind::Host;
    case memory::memory_kind::pinned:
        return protos::MemoryKind::Pinned;
    case memory::memory_kind::device:
        return protos::MemoryKind::Device;
    case memory::memory_kind::managed:
        return protos::MemoryKind::Managed;
    default:
        LOG(FATAL) << "unhandled mrc::memory::memory_kind";
    };

    return protos::MemoryKind::None;
}

}  // namespace mrc::codable
