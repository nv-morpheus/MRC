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

    static initializer_t tap(py_mirror_tap_t& self,
                             initializer_t initializer,
                             const std::string tap_from,
                             const std::string tap_to)
    {
        return self.tap(initializer, tap_from, tap_to);
    }

    static initializer_t stream_to(py_mirror_tap_t& self, initializer_t initializer, const std::string stream_to)
    {
        return self.stream_to(initializer, stream_to);
    }

    // TODO(create or extend ingress ports)
};

void init_mirror_tap_util(py::module_& module)
{
    using PythonMirrorTapUtil = mrc::modules::MirrorTapUtil<pymrc::PyHolder>;
    auto MirrorTap = py::class_<PythonMirrorTapUtil, std::shared_ptr<PythonMirrorTapUtil>>(module, "MirrorTap");

    MirrorTap.def(py::init(py::overload_cast<const std::string&>(&MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);
    MirrorTap.def(py::init(py::overload_cast<const std::string&, py::dict>(&MirrorTapUtilProxy::create)),
                  py::return_value_policy::take_ownership);

    MirrorTap.def("tap", &MirrorTapUtilProxy::tap, py::arg("initializer"), py::arg("tap_from"), py::arg("tap_to"));
    MirrorTap.def("stream_to", &MirrorTapUtilProxy::stream_to, py::arg("initializer"), py::arg("stream_to"));
}
}  // namespace mrc::pymrc