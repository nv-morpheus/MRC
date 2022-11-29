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

#include "mrc_qs_hybrid/data_object.hpp"

#include <glog/logging.h>
#include <mrc/utils/string_utils.hpp>
#include <pybind11/pybind11.h>

#include <memory>
#include <sstream>
#include <utility>

namespace mrc::quickstart::hybrid::common {
namespace py = pybind11;

PYBIND11_MODULE(data, m)
{
    m.doc() = R"pbdoc(
        -----------------------
        .. currentmodule:: quickstart
        .. autosummary::
           :toctree: _generate
            TODO(Documentation)
        )pbdoc";

    py::class_<DataObject, std::shared_ptr<DataObject>>(m, "DataObject")
        .def(py::init<>([](std::string name, int value) {
            // Create a new instance
            return std::make_shared<DataObject>(name, value);
        }))
        .def_readwrite("name", &DataObject::name)
        .def_readwrite("value", &DataObject::value)
        .def("__repr__",
             [](DataObject& self) {
                 return MRC_CONCAT_STR("{Name: '" << self.name << "', Value: " << self.value << "}");
             })
        .def(py::pickle(
            [](const DataObject& data_object) {  // __getstate__
                return py::make_tuple(data_object.name, data_object.value);
            },
            [](py::tuple pickled_data_object) {  // __setstate__
                if (pickled_data_object.size() != 2)
                {
                    throw std::runtime_error{"Invalid pickle state -- failed to restore object"};
                }
                DataObject data_object(pickled_data_object[0].cast<std::string>(), pickled_data_object[1].cast<int>());

                return data_object;
            }));

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace mrc::quickstart::hybrid::common
