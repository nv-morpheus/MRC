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

#include "pymrc/module_definitions/mirror_tap_util.hpp"

#include "pymrc/module_registry.hpp"
#include "pymrc/types.hpp"
#include "pymrc/utils.hpp"

#include "mrc/modules/mirror_tap/mirror_tap_util.hpp"
#include "mrc/modules/module_registry_util.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/version.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <string>

const std::vector<unsigned int> PybindSegmentModuleVersion{mrc_VERSION_MAJOR, mrc_VERSION_MINOR, mrc_VERSION_PATCH};

namespace py = pybind11;

namespace {
class MirrorTapUtilProxy
{
    using py_mirror_tap_t = mrc::modules::MirrorTapUtil<mrc::pymrc::PyHolder>;

  public:
    static std::shared_ptr<py_mirror_tap_t> create(const std::string& name)
    {
        return std::make_shared<py_mirror_tap_t>(name);
    }

    static std::shared_ptr<py_mirror_tap_t> create(const std::string& name, py::dict config)
    {
        return std::make_shared<py_mirror_tap_t>(name, mrc::pymrc::cast_from_pyobject(config));
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
}  // namespace

namespace mrc::pymrc {
[[maybe_unused]] void register_mirror_tap_modules()
{
    using namespace mrc::modules;

    ModelRegistryUtil::create_registered_module<mrc::modules::MirrorTapModule<pymrc::PyHolder>>(
        "MirrorTap",
        "mrc",
        PybindSegmentModuleVersion);

    ModelRegistryUtil::create_registered_module<MirrorTapStreamModule<pymrc::PyHolder>>("MirrorStreamBufferImmediate",
                                                                                        "mrc",
                                                                                        PybindSegmentModuleVersion);

    ModelRegistryUtil::create_registered_module<StreamBufferModule<pymrc::PyHolder>>("StreamBufferImmediate",
                                                                                     "mrc",
                                                                                     PybindSegmentModuleVersion);
}

[[maybe_unused]] void init_mirror_tap_util(py::module_& module)
{
    using python_mirror_tap_util_t = mrc::modules::MirrorTapUtil<pymrc::PyHolder>;
    auto MirrorTap = py::class_<python_mirror_tap_util_t, std::shared_ptr<python_mirror_tap_util_t>>(module,
                                                                                                     "MirrorTap");

    MirrorTap.def(py::init(py::overload_cast<const std::string&>(&::MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);

    MirrorTap.def(py::init(py::overload_cast<const std::string&, py::dict>(&::MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);

    MirrorTap.def("tap", &python_mirror_tap_util_t::tap, py::arg("initializer"), py::arg("tap_from"), py::arg("tap_to"));

    MirrorTap.def("stream_to", &python_mirror_tap_util_t::stream_to, py::arg("initializer"), py::arg("stream_to"));

    MirrorTap.def("get_ingress_tap_name", &python_mirror_tap_util_t::get_ingress_tap_name);

    MirrorTap.def("get_egress_tap_name", &python_mirror_tap_util_t::get_egress_tap_name);

    MirrorTap.def("create_or_extend_ingress_ports",
                  &::MirrorTapUtilProxy::create_or_extend_ingress_ports,
                  py::arg("ingress_ports"));

    MirrorTap.def("create_or_extend_egress_ports",
                  &::MirrorTapUtilProxy::create_or_extend_egress_ports,
                  py::arg("egress_ports"));
}
}  // namespace mrc::pymrc
