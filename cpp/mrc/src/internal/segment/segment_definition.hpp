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
#include "mrc/segment/segment.hpp"
#include "mrc/types.hpp"

#include <map>
#include <string>
#include <vector>

namespace mrc::segment {

class SegmentDefinition final : public ISegment
{
  public:
    SegmentDefinition(std::string name,
                      IngressPortsBase ingress_ports,
                      EgressPortsBase egress_ports,
                      segment_initializer_fn_t initializer);

    SegmentID id() const override;
    const std::string& name() const override;
    std::vector<std::string> ingress_port_names() const override;
    std::vector<std::string> egress_port_names() const override;

    const segment_initializer_fn_t& initializer_fn() const;

    const std::map<std::string, egress_initializer_t>& egress_initializers() const;
    const std::map<std::string, ingress_initializer_t>& ingress_initializers() const;

  private:
    void validate_ports() const;

    SegmentID m_id;
    std::string m_name;
    std::map<std::string, egress_initializer_t> m_egress_initializers;
    std::map<std::string, ingress_initializer_t> m_ingress_initializers;
    segment_initializer_fn_t m_initializer_fn;
};

}  // namespace mrc::segment
