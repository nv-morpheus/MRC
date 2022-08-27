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

#include "srf/memory/memory_kind.hpp"
#include "srf/protos/codable.pb.h"

namespace srf::codable {

// convert srf::memory::memory_kind enum to srf::codable::protos::MemoryKind
protos::MemoryKind encode_memory_type(memory::memory_kind mem_kind);

// convert srf::codable::protos::MemoryKind to srf::memory::memory_kind
memory::memory_kind decode_memory_type(const protos::MemoryKind& proto_kind);

}  // namespace srf::codable
