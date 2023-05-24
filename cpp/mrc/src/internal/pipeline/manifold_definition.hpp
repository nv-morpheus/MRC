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

#pragma once

#include "mrc/segment/initializers.hpp"
#include "mrc/types.hpp"

#include <map>
#include <memory>
#include <string>
#include <typeindex>

namespace mrc::segment {
class SegmentDefinition;
struct EgressPortsBase;
struct IngressPortsBase;
}  // namespace mrc::segment

namespace mrc::manifold {
class Interface;
}

namespace mrc::runnable {
class IRunnableResources;
}

namespace mrc::pipeline {

class ManifoldDefinition
{
  public:
    ManifoldDefinition(std::string name, std::type_index type_index, segment::manifold_initializer_fn_t initializer_fn);
    ~ManifoldDefinition();

    const std::string& name() const;

    std::type_index type_index() const;

    std::shared_ptr<manifold::Interface> build(runnable::IRunnableResources& resources) const;

  private:
    std::string m_name;
    std::type_index m_type_index;
    segment::manifold_initializer_fn_t m_initializer_fn;
};

}  // namespace mrc::pipeline
