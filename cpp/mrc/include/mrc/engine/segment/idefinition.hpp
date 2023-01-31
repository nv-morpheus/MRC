/**
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

#include <map>
#include <memory>
#include <string>

namespace mrc::internal::pipeline {
class IPipeline;
}

namespace mrc::internal::segment {

class Definition;

struct IDefinition
{
    IDefinition(std::string name,
                std::map<std::string, ::mrc::segment::ingress_initializer_t> ingress_initializers,
                std::map<std::string, ::mrc::segment::egress_initializer_t> egress_initializers,
                ::mrc::segment::backend_initializer_fn_t backend_initializer);
    virtual ~IDefinition() = 0;

    const std::string& name() const;

  private:
    std::shared_ptr<Definition> m_impl;
    friend pipeline::IPipeline;
};

}  // namespace mrc::internal::segment
