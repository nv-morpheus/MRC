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

#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/ingress_ports.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace srf::pysrf {

#define SRF_MAX_EGRESS_PORTS 10
#define SRF_MAX_INGRESS_PORTS 10

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class Pipeline
{
  public:
    Pipeline();

    /**
     * @brief Create a new SRF segment
     * @param name Segment name
     * @param init initializer used to define segment internals
     */
    void make_segment(const std::string& name, const std::function<void(srf::segment::Builder&)>& init);

    /**
     * @brief
     * @param name Segment name
     * @param ingress_port_ids Vector of strings with unique segment ingress port names
     *  note: these must also be unique with respect to egress port ids.
     * @param egress_port_ids Vector of strings with unique segment egress port names.
     *  note: these must also be unique with respect to ingress port ids.
     * @param init
     */
    void make_segment(const std::string& name,
                      const std::vector<std::string>& ingress_port_ids,
                      const std::vector<std::string>& egress_port_ids,
                      const std::function<void(srf::segment::Builder&)>& init);

    std::unique_ptr<srf::pipeline::Pipeline> swap();

  private:
    std::unique_ptr<srf::pipeline::Pipeline> m_pipeline;

    /**
     * @brief Used for runtime ingress/egress port construction. Assumes all port data types are py::objects
     */
    void dynamic_port_config(const std::string& name,
                             const std::vector<std::string>& ingress_port_ids,
                             const std::vector<std::string>& egress_port_ids,
                             const std::function<void(srf::segment::Builder&)>& init);

    void dynamic_port_config_ingress(const std::string& name,
                                     const std::vector<std::string>& ingress_port_ids,
                                     const std::function<void(srf::segment::Builder&)>& init);

    void dynamic_port_config_egress(const std::string& name,
                                    const std::vector<std::string>& egress_port_ids,
                                    const std::function<void(srf::segment::Builder&)>& init);

    template <typename... IngressPortTypesT>
    void typed_dynamic_port_config_egress(const std::string& name,
                                          const segment::IngressPorts<IngressPortTypesT...>& ingress_ports,
                                          const std::vector<std::string>& egress_port_ids,
                                          const std::function<void(srf::segment::Builder&)>& init);
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
