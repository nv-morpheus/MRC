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

#include "mrc/options/services.hpp"
#include "mrc/runnable/engine_factory.hpp"
#include "mrc/runnable/internal_service.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"

#include <map>
#include <memory>
#include <string>

namespace mrc::runnable {

/**
 * @brief Configuration object used to construct a LaunchControl object
 *
 */
struct LaunchControlConfig
{
    using resource_group_map_t = std::map<std::string, std::shared_ptr<EngineFactory>>;

    // ResourceGroups - resources used to build Engines
    resource_group_map_t resource_groups;

    // todo(#129)
    // map of segment ids to a user defiend launch option
    std::map<SegmentID, LaunchOptions> segment_options;

    // default options for all non-service runnables
    LaunchOptions default_options{};

    // service options from public api
    // ServiceOptions services;
};

}  // namespace mrc::runnable
