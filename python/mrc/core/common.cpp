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

#include "pymrc/edge_adapter.hpp"
#include "pymrc/port_builders.hpp"
#include "pymrc/types.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/manifold/egress.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <boost/fiber/future/future.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

// IWYU pragma: no_include <boost/fiber/future/detail/shared_state.hpp>
// IWYU pragma: no_include <boost/fiber/future/detail/task_base.hpp>
// IWYU pragma: no_include <boost/smart_ptr/detail/operator_bool.hpp>
// IWYU pragma: no_include <pybind11/detail/common.h>
// IWYU pragma: no_include "rx-includes.hpp"

namespace mrc::pymrc {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(common, module)
{
    module.doc() = R"pbdoc(
        Python bindings for MRC common functionality / utilities
        -------------------------------
        .. currentmodule:: common
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    EdgeAdapterUtil::register_data_adapters<PyHolder>();
    PortBuilderUtil::register_port_util<PyHolder>();

    module.attr("__version__") =
        MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "." << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
