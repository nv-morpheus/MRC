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

#include <glog/logging.h>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>
#include <utility>

namespace mrc::quickstart::hybrid::ex00_wrap_data_objects {
namespace py = pybind11;

struct MyDataObject
{
    MyDataObject(std::string n = "", int v = 0) : name(std::move(n)), value(v) {}
    std::string name;
    int value{0};
};

PYBIND11_MODULE(data, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: quickstart
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    py::class_<MyDataObject, std::shared_ptr<MyDataObject>>(m, "MyDataObject")
        .def(py::init<>([](std::string name, int value) {
            // Create a new instance
            return std::make_shared<MyDataObject>(name, value);
        }))
        .def_readwrite("name", &MyDataObject::name)
        .def_readwrite("value", &MyDataObject::value)
        .def("__repr__", [](MyDataObject& self) {
            return MRC_CONCAT_STR("{Name: '" << self.name << "', Value: " << self.value << "}");
        });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace mrc::quickstart::hybrid::ex00_wrap_data_objects
