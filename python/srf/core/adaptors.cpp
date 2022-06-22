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

#include <pysrf/edge_adaptor.hpp>
#include <pysrf/types.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace srf::pysrf {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(adaptors, m)
{
    m.doc() = R"pbdoc(
        Force register edge adaptors.
    )pbdoc";

    // Register pysrf adaptors
    node::EdgeAdaptorRegistry::register_source_adaptor(typeid(PyHolder),
                                                       PysrfEdgeAdapterUtil::build_source_adaptor<PyHolder>());
    node::EdgeAdaptorRegistry::register_sink_adaptor(typeid(PyHolder),
                                                     PysrfEdgeAdapterUtil::build_sink_adaptor<PyHolder>());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf