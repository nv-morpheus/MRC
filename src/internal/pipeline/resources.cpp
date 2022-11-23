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

#include "internal/pipeline/resources.hpp"

#include "mrc/metrics/registry.hpp"

#include <glog/logging.h>

namespace mrc::internal::pipeline {

Resources::Resources(resources::Manager& resources) :
  m_resources(resources),
  m_metrics_registry(std::make_unique<metrics::Registry>())
{}

resources::Manager& Resources::resources() const
{
    return m_resources;
}

metrics::Registry& Resources::metrics_registry() const
{
    DCHECK(m_metrics_registry);
    return *m_metrics_registry;
}

}  // namespace mrc::internal::pipeline
