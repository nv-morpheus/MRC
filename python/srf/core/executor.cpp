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

#include "pysrf/executor.hpp"

#include "pysrf/utils.hpp"

#include "srf/options/options.hpp"
#include "srf/version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>
#include <utility>  // for move
// IWYU thinks we need vectir for py::class_<Executor, std::shared_ptr<Executor>>
// IWYU pragma: no_include <vector>

namespace srf::pysrf {

namespace py = pybind11;

PYBIND11_MODULE(executor, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF executors
        -------------------------------
        .. currentmodule:: executor
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");
    pysrf::import(module, "srf.core.options");
    pysrf::import(module, "srf.core.pipeline");

    py::class_<Awaitable, std::shared_ptr<Awaitable>>(module, "Awaitable")
        .def(py::init<>())
        .def("__iter__", &Awaitable::iter)
        .def("__await__", &Awaitable::await)
        .def("__next__", &Awaitable::next);

    py::class_<Executor, std::shared_ptr<Executor>>(module, "Executor")
        .def(py::init<>([]() {
            auto options = std::make_shared<srf::Options>();

            auto* exec = new Executor(std::move(options));

            return exec;
        }))
        .def(py::init<>([](std::shared_ptr<srf::Options> options) {
            auto* exec = new Executor(std::move(options));

            return exec;
        }))
        .def("start", &Executor::start)
        .def("stop", &Executor::stop)
        .def("join", &Executor::join)
        .def("join_async", &Executor::join_async)
        .def("register_pipeline", &Executor::register_pipeline);

    py::class_<PyBoostFuture>(module, "Future")
        .def(py::init<>([]() { return PyBoostFuture(); }))
        .def("result", &PyBoostFuture::py_result)
        .def("set_result", &PyBoostFuture::set_result);

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf
