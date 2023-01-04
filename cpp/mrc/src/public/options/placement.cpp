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

#include "mrc/options/placement.hpp"

namespace mrc {

PlacementStrategy PlacementOptions::cpu_strategy() const
{
    return m_cpu_strategy;
}

PlacementResources PlacementOptions::resources_strategy() const
{
    return m_resources_strategy;
}
PlacementOptions& PlacementOptions::cpu_strategy(const PlacementStrategy& strategy)
{
    m_cpu_strategy = strategy;
    return *this;
}
PlacementOptions& PlacementOptions::resources_strategy(const PlacementResources& strategy)
{
    m_resources_strategy = strategy;
    return *this;
}
}  // namespace mrc
