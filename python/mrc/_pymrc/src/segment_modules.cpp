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

#include "pymrc/segment_modules.hpp"

#include "pymrc/utils.hpp"

#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/object.hpp"

#include <pybind11/pytypes.h>

#include <algorithm>
#include <memory>

namespace mrc::pymrc {
std::string SegmentModuleProxy::component_prefix(mrc::modules::SegmentModule& self)
{
    return self.component_prefix();
}

pybind11::dict SegmentModuleProxy::config(mrc::modules::SegmentModule& self)
{
    return cast_from_json(self.config());
}

const std::string& SegmentModuleProxy::name(mrc::modules::SegmentModule& self)
{
    return self.name();
}

std::string SegmentModuleProxy::module_type_name(mrc::modules::SegmentModule& self)
{
    return self.module_type_name();
}

std::vector<std::string> SegmentModuleProxy::input_ids(mrc::modules::SegmentModule& self)
{
    return self.input_ids();
}

std::vector<std::string> SegmentModuleProxy::output_ids(mrc::modules::SegmentModule& self)
{
    return self.output_ids();
}

std::shared_ptr<mrc::segment::ObjectProperties> SegmentModuleProxy::input_port(mrc::modules::SegmentModule& self,
                                                                               const std::string& input_id)
{
    return self.input_port(input_id);
}

const mrc::modules::SegmentModule::segment_module_port_map_t& SegmentModuleProxy::input_ports(
    mrc::modules::SegmentModule& self)
{
    return self.input_ports();
}

std::shared_ptr<mrc::segment::ObjectProperties> SegmentModuleProxy::output_port(mrc::modules::SegmentModule& self,
                                                                                const std::string& output_id)
{
    return self.output_port(output_id);
}

const mrc::modules::SegmentModule::segment_module_port_map_t& SegmentModuleProxy::output_ports(
    mrc::modules::SegmentModule& self)
{
    return self.output_ports();
}

}  // namespace mrc::pymrc
