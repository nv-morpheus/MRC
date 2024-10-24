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
#include "pymrc/utils.hpp"

#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/combine_latest.hpp"
#include "mrc/node/operators/round_robin_router_typeless.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/operators/with_latest_from.hpp"
#include "mrc/node/operators/zip.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/utils/tuple_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

    py::class_<mrc::segment::Object<node::ZipTypelessBase>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::ZipTypelessBase>>>(py_mod, "ZipComponent")
        .def(
            py::init<>([](mrc::segment::IBuilder& builder,
                          std::string name,
                          size_t count,
                          PyFuncHolder<py::object(py::tuple)> convert_fn) {
                std::function<PyObjectHolder(py::tuple)> convert_fn_wrapped =
                    [convert_fn = std::move(convert_fn)](py::tuple input_data) {
                        if (convert_fn)
                        {
                            return PyObjectHolder(convert_fn(std::move(input_data)));
                        }

                        return PyObjectHolder(std::move(input_data));
                    };

                auto make_node = [&builder,
                                  convert_fn_wrapped = std::move(convert_fn_wrapped)]<size_t N>(std::string name) {
                    return builder
                        .construct_object<
                            node::ZipTransformComponent<utils::repeat_tuple_type_t<PyObjectHolder, N>, PyObjectHolder>>(
                            name,
                            [convert_fn_wrapped = std::move(convert_fn_wrapped)](
                                utils::repeat_tuple_type_t<PyObjectHolder, N>&& input_data) {
                                py::gil_scoped_acquire gil;

                                return convert_fn_wrapped(py::cast(std::move(input_data)));
                            })
                        ->template as<node::ZipTypelessBase>();
                };

                if (count == 1)
                {
                    return make_node.template operator()<1>(name);
                }
                if (count == 2)
                {
                    return make_node.template operator()<2>(name);
                }
                if (count == 3)
                {
                    return make_node.template operator()<3>(name);
                }
                if (count == 4)
                {
                    return make_node.template operator()<4>(name);
                }

                throw std::runtime_error("Unsupported count!");
            }),
            py::arg("builder"),
            py::arg("name"),
            py::kw_only(),
            py::arg("count"),
            py::arg("convert_fn") = py::none())
        .def("get_sink", [](mrc::segment::Object<node::ZipTypelessBase>& self, size_t index) {
            return self.get_child(MRC_CONCAT_STR("sink[" << index << "]"));
        });

    py::class_<mrc::segment::Object<node::WithLatestFromTypelessBase>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::WithLatestFromTypelessBase>>>(py_mod,
                                                                                        "WithLatestFromComponent")
        .def(py::init<>([](mrc::segment::IBuilder& builder,
                           std::string name,
                           size_t count,
                           PyFuncHolder<py::object(py::tuple)> convert_fn) {
                 std::function<PyObjectHolder(py::tuple)> convert_fn_wrapped =
                     [convert_fn = std::move(convert_fn)](py::tuple input_data) {
                         if (convert_fn)
                         {
                             return PyObjectHolder(convert_fn(std::move(input_data)));
                         }

                         return PyObjectHolder(std::move(input_data));
                     };

                 auto make_node = [&builder,
                                   convert_fn_wrapped = std::move(convert_fn_wrapped)]<size_t N>(std::string name) {
                     return builder
                         .construct_object<
                             node::WithLatestFromTransformComponent<utils::repeat_tuple_type_t<PyObjectHolder, N>,
                                                                    PyObjectHolder>>(
                             name,
                             [convert_fn_wrapped = std::move(convert_fn_wrapped)](
                                 utils::repeat_tuple_type_t<PyObjectHolder, N>&& input_data) {
                                 py::gil_scoped_acquire gil;

                                 return convert_fn_wrapped(py::cast(std::move(input_data)));
                             })
                         ->template as<node::WithLatestFromTypelessBase>();
                 };

                 if (count == 1)
                 {
                     return make_node.template operator()<1>(name);
                 }
                 if (count == 2)
                 {
                     return make_node.template operator()<2>(name);
                 }
                 if (count == 3)
                 {
                     return make_node.template operator()<3>(name);
                 }
                 if (count == 4)
                 {
                     return make_node.template operator()<4>(name);
                 }

                 throw std::runtime_error("Unsupported count!");
             }),
             py::arg("builder"),
             py::arg("name"),
             py::kw_only(),
             py::arg("count"),
             py::arg("convert_fn") = py::none())
        .def("get_sink", [](mrc::segment::Object<node::WithLatestFromTypelessBase>& self, size_t index) {
            return self.get_child(MRC_CONCAT_STR("sink[" << index << "]"));
        });

    py::class_<mrc::segment::Object<node::CombineLatestTypelessBase>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::CombineLatestTypelessBase>>>(py_mod, "CombineLatestComponent")
        .def(py::init<>([](mrc::segment::IBuilder& builder,
                           std::string name,
                           size_t count,
                           PyFuncHolder<py::object(py::tuple)> convert_fn) {
                 std::function<PyObjectHolder(py::tuple)> convert_fn_wrapped =
                     [convert_fn = std::move(convert_fn)](py::tuple input_data) {
                         if (convert_fn)
                         {
                             return PyObjectHolder(convert_fn(std::move(input_data)));
                         }

                         return PyObjectHolder(std::move(input_data));
                     };

                 auto make_node = [&builder,
                                   convert_fn_wrapped = std::move(convert_fn_wrapped)]<size_t N>(std::string name) {
                     return builder
                         .construct_object<
                             node::CombineLatestTransformComponent<utils::repeat_tuple_type_t<PyObjectHolder, N>,
                                                                   PyObjectHolder>>(
                             name,
                             [convert_fn_wrapped = std::move(convert_fn_wrapped)](
                                 utils::repeat_tuple_type_t<PyObjectHolder, N>&& input_data) {
                                 py::gil_scoped_acquire gil;

                                 return convert_fn_wrapped(py::cast(std::move(input_data)));
                             })
                         ->template as<node::CombineLatestTypelessBase>();
                 };

                 if (count == 1)
                 {
                     return make_node.template operator()<1>(name);
                 }
                 if (count == 2)
                 {
                     return make_node.template operator()<2>(name);
                 }
                 if (count == 3)
                 {
                     return make_node.template operator()<3>(name);
                 }
                 if (count == 4)
                 {
                     return make_node.template operator()<4>(name);
                 }

                 throw std::runtime_error("Unsupported count!");
             }),
             py::arg("builder"),
             py::arg("name"),
             py::kw_only(),
             py::arg("count"),
             py::arg("convert_fn") = py::none())
        .def("get_sink", [](mrc::segment::Object<node::CombineLatestTypelessBase>& self, size_t index) {
            return self.get_child(MRC_CONCAT_STR("sink[" << index << "]"));
        });

    py::class_<mrc::segment::Object<node::LambdaStaticRouterComponent<std::string, PyObjectHolder>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::LambdaStaticRouterComponent<std::string, PyObjectHolder>>>>(
        py_mod,
        "RouterComponent")
        .def(py::init<>([](mrc::segment::IBuilder& builder,
                           std::string name,
                           std::vector<std::string> router_keys,
                           OnDataFunction key_fn) {
                 return builder.construct_object<node::LambdaStaticRouterComponent<std::string, PyObjectHolder>>(
                     name,
                     router_keys,
                     [key_fn_cap = std::move(key_fn)](const PyObjectHolder& data) -> std::string {
                         py::gil_scoped_acquire gil;

                         auto ret_key     = key_fn_cap(data.copy_obj());
                         auto ret_key_str = py::str(ret_key);

                         return std::string(ret_key_str);
                     });
             }),
             py::arg("builder"),
             py::arg("name"),
             py::kw_only(),
             py::arg("router_keys"),
             py::arg("key_fn"))
        .def(
            "get_source",
            [](mrc::segment::Object<node::LambdaStaticRouterComponent<std::string, PyObjectHolder>>& self,
               py::object key) {
                std::string key_str = py::str(key);

                return self.get_child(key_str);
            },
            py::arg("key"));

    py::class_<mrc::segment::Object<node::LambdaStaticRouterRunnable<std::string, PyObjectHolder>>,
               mrc::segment::ObjectProperties,
               std::shared_ptr<mrc::segment::Object<node::LambdaStaticRouterRunnable<std::string, PyObjectHolder>>>>(
        py_mod,
        "Router")
        .def(py::init<>([](mrc::segment::IBuilder& builder,
                           std::string name,
                           std::vector<std::string> router_keys,
                           OnDataFunction key_fn) {
                 return builder.construct_object<node::LambdaStaticRouterRunnable<std::string, PyObjectHolder>>(
                     name,
                     router_keys,
                     [key_fn_cap = std::move(key_fn)](const PyObjectHolder& data) -> std::string {
                         py::gil_scoped_acquire gil;

                         auto ret_key     = key_fn_cap(data.copy_obj());
                         auto ret_key_str = py::str(ret_key);

                         return std::string(ret_key_str);
                     });
             }),
             py::arg("builder"),
             py::arg("name"),
             py::kw_only(),
             py::arg("router_keys"),
             py::arg("key_fn"))
        .def(
            "get_source",
            [](mrc::segment::Object<node::LambdaStaticRouterRunnable<std::string, PyObjectHolder>>& self,
               py::object key) {
                std::string key_str = py::str(key);

                return self.get_child(key_str);
            },
            py::arg("key"));

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
