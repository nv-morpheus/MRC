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

#include "pysrf/edge_adapter.hpp"
#include "pysrf/port_builders.hpp"

#include "srf/node/edge_adapter_registry.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace srf::pysrf {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(common, m)
{
    m.doc() = R"pbdoc(
        Python bindings for SRF common functionality / utilities
        -------------------------------
        .. currentmodule:: common
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    node::EdgeAdapterRegistry::register_source_adapter(typeid(PyHolder),
                                                       EdgeAdapterUtil::build_source_adapter<PyHolder>());

    node::EdgeAdapterRegistry::register_sink_adapter(typeid(PyHolder),
                                                     EdgeAdapterUtil::build_sink_adapter<PyHolder>());

    PortUtilBuilder::register_port_util<PyHolder>();
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf
