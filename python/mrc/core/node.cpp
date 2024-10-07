/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/node.hpp"

#include "pymrc/utilities/function_wrappers.hpp"
#include "pymrc/utilities/object_wrappers.hpp"
#include "pymrc/utils.hpp"

#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/round_robin_router_typeless.hpp"
#include "mrc/node/operators/zip.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>
#include <sstream>
#include <string>

namespace mrc::pymrc {
namespace py = pybind11;

PYBIND11_MODULE(node, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC nodes
        -------------------------------
        .. currentmodule:: node
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(py_mod, "mrc.core.common");
    pymrc::import(py_mod, "mrc.core.segment");  // Needed for Builder and SegmentObject

    py::class_<mrc::segment::Object<node::BroadcastTypeless>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::BroadcastTypeless>>>(py_mod, "Broadcast")
        .def(py::init<>([](mrc::segment::IBuilder& builder, std::string name) {
            auto node = builder.construct_object<node::BroadcastTypeless>(name);

            return node;
        }));

    py::class_<mrc::segment::Object<node::RoundRobinRouterTypeless>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::RoundRobinRouterTypeless>>>(py_mod, "RoundRobinRouter")
        .def(py::init<>([](mrc::segment::IBuilder& builder, std::string name) {
            auto node = builder.construct_object<node::RoundRobinRouterTypeless>(name);

            return node;
        }));

    py::class_<mrc::segment::Object<node::ZipBase>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::ZipBase>>>(py_mod, "Zip")
        .def(py::init<>([](mrc::segment::IBuilder& builder, std::string name, size_t count) {
            // std::shared_ptr<mrc::segment::ObjectProperties> node;

            if (count == 2)
            {
                return builder.construct_object<node::Zip<PyObjectHolder, PyObjectHolder>>(name)->as<node::ZipBase>();
            }
            else
            {
                py::print("Unsupported count!");
                throw std::runtime_error("Unsupported count!");
            }
        }))
        .def("get_sink", [](mrc::segment::Object<node::ZipBase>& self, size_t index) {
            return self.get_child(MRC_CONCAT_STR("sink[" << index << "]"));
        });

    // py::class_<mrc::segment::Object<node::ZipBase<std::tuple<int>>>,
    //            mrc::segment::ObjectProperties,
    //            std::shared_ptr<mrc::segment::Object<node::ZipBase>>>(py_mod, "Zip")
    //     .def(py::init<>([](mrc::segment::IBuilder& builder, std::string name, size_t count) {
    //         // std::shared_ptr<mrc::segment::ObjectProperties> node;

    //         if (count == 2)
    //         {
    //             return builder.construct_object<node::Zip<PyObjectHolder,
    //             PyObjectHolder>>(name)->as<node::ZipBase>();
    //         }
    //         else
    //         {
    //             py::print("Unsupported count!");
    //             throw std::runtime_error("Unsupported count!");
    //         }
    //     }))
    //     .def("get_sink", [](mrc::segment::Object<node::ZipBase>& self, size_t index) {
    //         return self.get_child(MRC_CONCAT_STR("sink[" << index << "]"));
    //     });

    py::class_<mrc::segment::Object<node::DynamicRouter<std::string, py::object>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::DynamicRouter<std::string, py::object>>>>(py_mod, "Router")
        .def(py::init<>([](mrc::segment::IBuilder& builder,
                           std::string name,
                           std::vector<std::string> router_keys,
                           OnDataFunction key_fn) {
                 return builder.construct_object<node::DynamicRouter<std::string, py::object>>(
                     name,
                     router_keys,
                     [key_fn_cap = std::move(key_fn)](const py::object& data) -> std::string {
                         return std::string(py::str(key_fn_cap(data)));
                     });
             }),
             py::arg("builder"),
             py::arg("name"),
             py::kw_only(),
             py::arg("router_keys"),
             py::arg("key_fn"))
        .def(
            "get_source",
            [](mrc::segment::Object<node::DynamicRouter<std::string, py::object>>& self, py::object key) {
                std::string key_str = py::str(key);

                return self.get_child(MRC_CONCAT_STR("source[" << key_str << "]"));
            },
            py::arg("key"));

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
