/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/pipeline/manifold_instance.hpp"

#include "internal/pipeline/manifold_definition.hpp"
#include "internal/resources/system_resources.hpp"
#include "internal/runtime/runtime.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/runnable/runnable_resources.hpp"
#include "mrc/utils/string_utils.hpp"

namespace mrc::pipeline {

ManifoldInstance::ManifoldInstance(runtime::Runtime& runtime,
                                   std::shared_ptr<const ManifoldDefinition> definition,
                                   uint64_t instance_id) :
  AsyncService(MRC_CONCAT_STR("Manifold[" << definition->name() << "]")),
  runnable::RunnableResourcesProvider(runtime),
  m_runtime(runtime),
  m_definition(std::move(definition)),
  m_instance_id(instance_id)
{}

ManifoldInstance::~ManifoldInstance() = default;

std::shared_ptr<manifold::Interface> ManifoldInstance::get_interface() const
{
    CHECK(m_interface) << "Must start ManifoldInstance before using the interface";

    return m_interface;
}

void ManifoldInstance::do_service_start(std::stop_token stop_token)
{
    m_interface = m_definition->build(m_runtime.resources().runnable());

    m_interface->start();

    this->mark_started();

    m_interface->join();
}

}  // namespace mrc::pipeline
