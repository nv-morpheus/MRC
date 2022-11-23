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

#include "internal/resources/manager.hpp"

#include "mrc/metrics/registry.hpp"

#include <memory>

namespace mrc::internal::pipeline {

class Resources
{
  public:
    Resources(resources::Manager& resources);

    resources::Manager& resources() const;
    metrics::Registry& metrics_registry() const;

  private:
    resources::Manager& m_resources;
    std::unique_ptr<metrics::Registry> m_metrics_registry;
};

}  // namespace mrc::internal::pipeline
