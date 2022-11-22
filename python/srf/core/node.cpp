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

#include "pysrf/node.hpp"

#include "pysrf/types.hpp"
#include "pysrf/utils.hpp"

#include "srf/node/operators/broadcast.hpp"
#include "srf/runnable/launch_options.hpp"
#include "srf/segment/builder.hpp"
#include "srf/segment/object.hpp"
#include "srf/utils/string_utils.hpp"
#include "srf/version.hpp"

#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>

namespace srf::pysrf {
namespace py = pybind11;

PYBIND11_MODULE(node, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF nodes
        -------------------------------
        .. currentmodule:: node
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");
    pysrf::import(module, "srf.core.segment");  // Needed for Builder and SegmentObject

    // py::class_<srf::segment::Object<PythonNode<PyHolder, PyHolder>>,
    //            srf::segment::ObjectProperties,
    //            std::shared_ptr<srf::segment::Object<PythonNode<PyHolder, PyHolder>>>>(m, "Node")
    //     .def(py::init<>([](srf::segment::Builder& builder, std::string name) {
    //         auto node = builder.construct_object<node::BroadcastTypeless>(name);

    //         return node;
    //     }));

    py::class_<srf::segment::Object<node::BroadcastTypeless>,
               srf::segment::ObjectProperties,
               std::shared_ptr<srf::segment::Object<node::BroadcastTypeless>>>(module, "Broadcast")
        .def(py::init<>([](srf::segment::Builder& builder, std::string name) {
            auto node = builder.construct_object<node::BroadcastTypeless>(name);

            return node;
        }));

    module.attr("__version__") =
        SRF_CONCAT_STR(srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH);
}
}  // namespace srf::pysrf
