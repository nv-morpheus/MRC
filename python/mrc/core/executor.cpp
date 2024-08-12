/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/executor.hpp"

#include "pymrc/pipeline.hpp"                     // IWYU pragma: keep
#include "pymrc/utilities/function_wrappers.hpp"  // for PyFuncWrapper
#include "pymrc/utils.hpp"

#include "mrc/options/options.hpp"
#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <memory>
#include <sstream>
#include <utility>  // for move

// IWYU thinks we need vectir for py::class_<Executor, std::shared_ptr<Executor>>
// IWYU pragma: no_include <vector>

namespace mrc::pymrc {

namespace py = pybind11;

PYBIND11_MODULE(executor, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC executors
        -------------------------------
        .. currentmodule:: executor
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(py_mod, "mrc.core.common");
    pymrc::import(py_mod, "mrc.core.options");
    pymrc::import(py_mod, "mrc.core.pipeline");

    py::class_<Awaitable, std::shared_ptr<Awaitable>>(py_mod, "Awaitable")
        .def(py::init<>())
        .def("__iter__", &Awaitable::iter)
        .def("__await__", &Awaitable::await)
        .def("__next__", &Awaitable::next);

    py::class_<Executor, std::shared_ptr<Executor>>(py_mod, "Executor")
        .def(py::init<>([]() {
            auto options = std::make_shared<mrc::Options>();

            auto* exec = new Executor(std::move(options));

            return exec;
        }))
        // .def(py::init(py::overload_cast<int>(&test_fn)))
        // .def(py::init<>(&test_fn, py::arg("options"), py::arg("state_change_cb") = py::none()))
        .def(py::init<>([](std::shared_ptr<mrc::Options> options, py::object state_change_cb) {
                 std::function<void(State)> wrapped_state_change_cb = nullptr;
                 if (!state_change_cb.is_none())
                 {
                     auto wrapped_cb         = PyFuncWrapper(std::move(state_change_cb));
                     wrapped_state_change_cb = [wrapped_cb = std::move(wrapped_cb)](State state) {
                         wrapped_cb.operator()<void, pybind11::int_>(static_cast<int>(state));
                     };
                 }

                 auto exec = std::make_shared<Executor>(std::move(options), std::move(wrapped_state_change_cb));

                 return exec;
             }),
             py::arg("options"),
             py::arg("state_change_cb") = py::none())
        .def("start", &Executor::start)
        .def("stop", &Executor::stop)
        .def("join", &Executor::join)
        .def("join_async", &Executor::join_async)
        .def("register_pipeline", &Executor::register_pipeline);

    py::class_<PyBoostFuture>(py_mod, "Future")
        .def(py::init<>([]() {
            return PyBoostFuture();
        }))
        .def("result", &PyBoostFuture::py_result)
        .def("set_result", &PyBoostFuture::set_result);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
