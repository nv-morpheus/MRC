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

#include "mrc/constants.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/runnable/types.hpp"

#include <cstdint>
#include <string>

namespace mrc::runnable {

struct LaunchOptions
{
    LaunchOptions() = default;
    LaunchOptions(std::string name, std::size_t pes = 1, std::size_t epps = 1) :
      engine_factory_name(std::move(name)),
      pe_count(pes),
      engines_per_pe(epps)
    {}

    std::size_t pe_count{1};
    std::size_t engines_per_pe{1};
    std::string engine_factory_name{default_engine_factory_name()};
};

struct ServiceLaunchOptions : public LaunchOptions
{
    int m_priority{MRC_DEFAULT_FIBER_PRIORITY};
};

}  // namespace mrc::runnable
