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

#include "mrc/types.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace mrc::segment {

class EgressPortBase;
class IngressPortBase;

}  // namespace mrc::segment

namespace mrc::internal::pipeline {
class IPipeline;
}

namespace mrc::internal::segment {

class Definition;
class IBuilder;

struct IDefinition
{
    using backend_initializer_fn_t = std::function<void(IBuilder&)>;
    using egress_initializer_t = std::function<std::shared_ptr<::mrc::segment::EgressPortBase>(const SegmentAddress&)>;
    using ingress_initializer_t =
        std::function<std::shared_ptr<::mrc::segment::IngressPortBase>(const SegmentAddress&)>;

    IDefinition(std::string name,
                std::map<std::string, ingress_initializer_t> ingress_initializers,
                std::map<std::string, egress_initializer_t> egress_initializers,
                backend_initializer_fn_t backend_initializer);
    virtual ~IDefinition() = 0;

    const std::string& name() const;

    // const SegmentID& id() const;
    // std::vector<std::string> ingress_port_names() const;
    // std::vector<std::string> egress_port_names() const;
    // [[deprecated]] std::string info() const;
    // [[deprecated]] std::string info(SegmentRank rank) const;

  private:
    std::shared_ptr<Definition> m_impl;
    friend pipeline::IPipeline;
};

}  // namespace mrc::internal::segment
