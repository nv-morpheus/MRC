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

#include "pymrc/utils.hpp"

#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/round_robin_router_typeless.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

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

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
