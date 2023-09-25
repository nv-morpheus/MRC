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

#include "segment_modules.hpp"

#include "pymrc/segment_modules.hpp"

#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/builder.hpp"  // IWYU pragma: keep

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <memory>

namespace mrc::pymrc {

namespace py = pybind11;

void PySegmentModule::initialize(segment::IBuilder& builder)
{
    PYBIND11_OVERLOAD_PURE(void, mrc::modules::SegmentModule, initialize, builder);
}

std::string PySegmentModule::module_type_name() const
{
    PYBIND11_OVERLOAD_PURE(std::string, mrc::modules::SegmentModule, module_type_name);
}

void init_segment_modules(py::module_& smodule)
{
    auto SegmentModule =
        py::class_<mrc::modules::SegmentModule, PySegmentModule, std::shared_ptr<mrc::modules::SegmentModule>>(smodule,
                                                                                                               "Segment"
                                                                                                               "Modul"
                                                                                                               "e");

    /** Segment Module Interface Declarations **/
    SegmentModule.def(py::init<std::string>());

    SegmentModule.def("config", &SegmentModuleProxy::config);

    SegmentModule.def("component_prefix", &SegmentModuleProxy::component_prefix);

    SegmentModule.def("input_port", &SegmentModuleProxy::input_port, py::arg("input_id"));

    SegmentModule.def("input_ports", &SegmentModuleProxy::input_ports);

    SegmentModule.def("module_type_name", &SegmentModuleProxy::module_type_name);

    SegmentModule.def("name", &SegmentModuleProxy::name);

    SegmentModule.def("output_port", &SegmentModuleProxy::output_port, py::arg("output_id"));

    SegmentModule.def("output_ports", &SegmentModuleProxy::output_ports);

    SegmentModule.def("input_ids", &SegmentModuleProxy::input_ids);

    SegmentModule.def("output_ids", &SegmentModuleProxy::output_ids);
}
}  // namespace mrc::pymrc
