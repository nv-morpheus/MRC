/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runnable/resources.hpp"
#include "internal/system/partition_provider.hpp"

#include "srf/utils/macros.hpp"

#include <cstddef>
#include <functional>

namespace srf::internal::resources {

/**
 * @brief Provider of both a system::PartitionProvider and a runnable::Resource assigned to the particular partition
 *
 * This provider avoids significant code duplication since virtually every type of of partition resource need both
 * partition information as well an the runnable resources.
 */
class PartitionResourceBase : public system::PartitionProvider
{
  public:
    PartitionResourceBase(runnable::Resources& runnable, std::size_t partition_id);

    runnable::Resources& runnable();

  private:
    std::reference_wrapper<runnable::Resources> m_runnable;
};

}  // namespace srf::internal::resources
