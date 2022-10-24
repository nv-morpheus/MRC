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

#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/forward.hpp"
#include "srf/segment/object.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class SegmentModuleProxy
{
  public:
    static std::string component_prefix(srf::modules::SegmentModule& self)
    {
        return self.component_prefix();
    }

    static pybind11::dict config(srf::modules::SegmentModule& self)
    {
        return cast_from_json(self.config());
    }

    static const std::string& name(srf::modules::SegmentModule& self)
    {
        return self.name();
    }

    static std::string module_name(srf::modules::SegmentModule& self)
    {
        return self.module_name();
    }

    static std::vector<std::string> input_ids(srf::modules::SegmentModule& self)
    {
        return self.input_ids();
    }

    static std::vector<std::string> output_ids(srf::modules::SegmentModule& self)
    {
        return self.output_ids();
    }

    static std::shared_ptr<srf::segment::ObjectProperties> input_port(srf::modules::SegmentModule& self,
                                                                      const std::string& input_id)
    {
        return self.input_port(input_id);
    }

    static const srf::modules::SegmentModule::segment_module_port_map_t& input_ports(srf::modules::SegmentModule& self)
    {
        return self.input_ports();
    }

    static std::shared_ptr<srf::segment::ObjectProperties> output_port(srf::modules::SegmentModule& self,
                                                                       const std::string& output_id)
    {
        return self.output_port(output_id);
    }

    static const srf::modules::SegmentModule::segment_module_port_map_t& output_ports(srf::modules::SegmentModule& self)
    {
        return self.output_ports();
    }
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
