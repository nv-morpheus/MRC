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

#include "pymrc/module_definitions/mirror_tap_util.hpp"

#include "pymrc/types.hpp"
#include "pymrc/utils.hpp"

#include "mrc/modules/mirror_tap/mirror_tap_util.hpp"
#include "mrc/segment/builder.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

namespace mrc::pymrc {

namespace py = pybind11;

class PySegmentModule : public mrc::modules::SegmentModule
{
    using mrc::modules::SegmentModule::SegmentModule;

    void initialize(segment::Builder& builder) override
    {
        PYBIND11_OVERLOAD_PURE(void, mrc::modules::SegmentModule, initialize, builder);
    }

    std::string module_type_name() const override
    {
        PYBIND11_OVERLOAD_PURE(std::string, mrc::modules::SegmentModule, module_type_name);
    }
};

class MirrorTapUtilProxy
{
    using initializer_t   = std::function<void(segment::Builder& builder)>;
    using py_mirror_tap_t = mrc::modules::MirrorTapUtil<pymrc::PyHolder>;

  public:
    static std::shared_ptr<py_mirror_tap_t> create(const std::string& name)
    {
        return std::make_shared<py_mirror_tap_t>(name);
    }

    static std::shared_ptr<py_mirror_tap_t> create(const std::string& name, py::dict config)
    {
        return std::make_shared<py_mirror_tap_t>(name, cast_from_pyobject(config));
    }

    static py::list create_or_extend_ingress_ports(py_mirror_tap_t& self, py::list ingress_ports)
    {
        ingress_ports.append(self.get_ingress_tap_name());
        return ingress_ports;
    }

    static py::list create_or_extend_egress_ports(py_mirror_tap_t& self, py::list ingress_ports)
    {
        ingress_ports.append(self.get_ingress_tap_name());
        return ingress_ports;
    }
};

void init_mirror_tap_util(py::module_& module)
{
    using PythonMirrorTapUtil = mrc::modules::MirrorTapUtil<pymrc::PyHolder>;
    auto MirrorTap = py::class_<PythonMirrorTapUtil, std::shared_ptr<PythonMirrorTapUtil>>(module, "MirrorTap");

    MirrorTap.def(py::init(py::overload_cast<const std::string&>(&MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);

    MirrorTap.def(py::init(py::overload_cast<const std::string&, py::dict>(&MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);

    MirrorTap.def("tap", &PythonMirrorTapUtil::tap, py::arg("initializer"), py::arg("tap_from"), py::arg("tap_to"));

    MirrorTap.def("stream_to", &PythonMirrorTapUtil::stream_to, py::arg("initializer"), py::arg("stream_to"));

    MirrorTap.def("get_ingress_tap_name", &PythonMirrorTapUtil::get_ingress_tap_name);

    MirrorTap.def("get_egress_tap_name", &PythonMirrorTapUtil::get_egress_tap_name);

    MirrorTap.def("create_or_extend_ingress_ports",
                  &MirrorTapUtilProxy::create_or_extend_ingress_ports,
                  py::arg("ingress_ports"));

    MirrorTap.def("create_or_extend_egress_ports",
                  &MirrorTapUtilProxy::create_or_extend_egress_ports,
                  py::arg("egress_ports"));
}
}  // namespace mrc::pymrc