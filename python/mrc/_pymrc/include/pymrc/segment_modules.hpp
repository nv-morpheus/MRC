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

#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/object.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>
#include <string>
#include <vector>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class SegmentModuleProxy
{
  public:
    static std::string component_prefix(mrc::modules::SegmentModule& self);

    static pybind11::dict config(mrc::modules::SegmentModule& self);

    static const std::string& name(mrc::modules::SegmentModule& self);

    static std::string module_type_name(mrc::modules::SegmentModule& self);

    static std::vector<std::string> input_ids(mrc::modules::SegmentModule& self);

    static std::vector<std::string> output_ids(mrc::modules::SegmentModule& self);

    static std::shared_ptr<mrc::segment::ObjectProperties> input_port(mrc::modules::SegmentModule& self,
                                                                      const std::string& input_id);

    static const mrc::modules::SegmentModule::segment_module_port_map_t& input_ports(mrc::modules::SegmentModule& self);

    static std::shared_ptr<mrc::segment::ObjectProperties> output_port(mrc::modules::SegmentModule& self,
                                                                       const std::string& output_id);

    static const mrc::modules::SegmentModule::segment_module_port_map_t& output_ports(
        mrc::modules::SegmentModule& self);
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
