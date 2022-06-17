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

#include "internal/runnable/resources.hpp"
#include "internal/system/resources.hpp"
#include "internal/system/system_provider.hpp"

#include <memory>

namespace srf::internal::resources {

class Manager final : public system::SystemProvider
{
  public:
    Manager(const system::SystemProvider& system);
    Manager(std::unique_ptr<system::Resources> resources);

    std::size_t device_count() const;
    std::size_t partition_count() const;

    runnable::Resources& runnable(std::size_t partition_id);

  private:
    std::unique_ptr<system::Resources> m_system;
    std::vector<runnable::Resources> m_runnable;
};

}  // namespace srf::internal::resources
