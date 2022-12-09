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

#include "mrc/core/bitmap.hpp"
#include "mrc/forward.hpp"

#include <cstdint>

namespace mrc {

struct PlacementGroup
{
    [[nodiscard]] virtual std::size_t id() const           = 0;
    [[nodiscard]] virtual const CpuSet& cpu_set() const    = 0;
    [[nodiscard]] virtual const NumaSet& numa_set() const  = 0;
    [[nodiscard]] virtual const Bitmap& gpu_set() const    = 0;
    [[nodiscard]] virtual std::size_t total_memory() const = 0;
    [[nodiscard]] virtual bool has_gpus() const            = 0;
};

}  // namespace mrc
