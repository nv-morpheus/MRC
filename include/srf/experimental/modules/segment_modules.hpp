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

#include "nlohmann/json.hpp"

#include "srf/segment/object.hpp"

#include <map>
#include <string>
#include <utility>
#include <vector>

#pragma once

namespace srf::segment {
class Builder;
}

namespace srf::modules {

class SegmentModulePort {
  public:
    template<typename AsTypeT>
    AsTypeT as() {
        return m_port_object->sink_typed<AsTypeT>;
    }

  private:
    std::shared_ptr<segment::ObjectProperties> m_port_object;
};

class SegmentModule
{
  public:
    friend segment::Builder;

    using SegmentModulePortT        = std::shared_ptr<segment::ObjectProperties>;
    using segment_module_port_map_t = std::map<std::string, std::shared_ptr<segment::ObjectProperties>>;

    SegmentModule() = delete;
    SegmentModule(std::string module_name);

    const std::string& name() const;
    const std::string& component_prefix() const;

    /**
     * Retrieve vector of input names -- these are only understood by the SegmentModule,
     * @return
     */
    virtual const std::vector<std::string> inputs() const  = 0;
    virtual const std::vector<std::string> outputs() const = 0;

    /**
     * Return a set of ObjectProperties for module inputs
     * @return
     */
    virtual segment_module_port_map_t input_ports() = 0;
    virtual SegmentModulePortT input_ports(const std::string& input_name) = 0;

    /**
     *
     * @return
     */
    virtual segment_module_port_map_t output_ports() = 0;
    virtual SegmentModulePortT output_ports(const std::string& output_name) = 0;

    /**
     *
     * @param builder
     */
    virtual void initialize(segment::Builder& builder) = 0;

    /**
     * Entrypoint for module constructor during build -- this lets us act like a std::function
     * @param builder
     */
    void operator()(segment::Builder& builder)
    {
        this->initialize(builder);
    };

  private:
    std::string m_module_name;
    std::string m_component_prefix;
};


}  // namespace srf::modules