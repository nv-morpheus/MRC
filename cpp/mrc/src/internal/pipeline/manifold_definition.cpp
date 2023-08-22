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

#include "internal/pipeline/manifold_definition.hpp"

namespace mrc::runnable {

ManifoldDefinition::ManifoldDefinition(std::string name,
                                       std::type_index type_index,
                                       segment::manifold_initializer_fn_t initializer_fn) :
  m_name(std::move(name)),
  m_type_index(std::move(type_index)),
  m_initializer_fn(std::move(initializer_fn))
{}

ManifoldDefinition::~ManifoldDefinition() = default;

const std::string& ManifoldDefinition::name() const
{
    return m_name;
}

std::type_index ManifoldDefinition::type_index() const
{
    return m_type_index;
}

std::shared_ptr<manifold::Interface> ManifoldDefinition::build(runnable::IRunnableResources& resources) const
{
    return m_initializer_fn(m_name, resources);
}

}  // namespace mrc::runnable
