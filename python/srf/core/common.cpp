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
#include "pysrf/types.hpp"

#include "srf/node/sink_properties.hpp"
#include "srf/version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <memory>
#include <vector>

// IWYU pragma: no_include <boost/fiber/future/detail/shared_state.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/task_base.hpp>
// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rx-includes.hpp"

namespace srf::pysrf {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(common, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF common functionality / utilities
        -------------------------------
        .. currentmodule:: common
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    EdgeAdapterUtil::register_data_adapters<PyHolder>();
    PortBuilderUtil::register_port_util<PyHolder>();

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf
