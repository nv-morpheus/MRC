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

#include "mrc/engine/segment/idefinition.hpp"

#include "internal/segment/definition.hpp"

#include <utility>

namespace mrc::internal::segment {

IDefinition::IDefinition(std::string name,
                         std::map<std::string, ingress_initializer_t> ingress_initializers,
                         std::map<std::string, egress_initializer_t> egress_initializers,
                         backend_initializer_fn_t backend_initializer) :
  m_impl(std::make_shared<Definition>(
      std::move(name), std::move(ingress_initializers), std::move(egress_initializers), std::move(backend_initializer)))
{}
IDefinition::~IDefinition() = default;

const std::string& IDefinition::name() const
{
    return m_impl->name();
}
}  // namespace mrc::internal::segment
