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

#include "pymrc/subscriber.hpp"

#include "pymrc/types.hpp"  // for PyObjectObserver, PyObjectSubscriber, PyObjectObservable, PySubscription
#include "pymrc/utils.hpp"

#include "mrc/utils/string_utils.hpp"
#include "mrc/version.hpp"

#include <pybind11/attr.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>         // IWYU pragma: keep
#include <pybind11/pybind11.h>
#include <rxcpp/rx.hpp>

#include <memory>
#include <sstream>

namespace mrc::pymrc {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(subscriber, py_mod)
{
    py_mod.doc() = R"pbdoc(
        Python bindings for MRC subscribers
        -------------------------------
        .. currentmodule:: subscriber
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pymrc::import(py_mod, "mrc.core.common");

    py::class_<PySubscription>(py_mod, "Subscription");

    py::class_<PyObjectObserver>(py_mod, "Observer")
        .def("on_next",
             &ObserverProxy::on_next,
             py::call_guard<py::gil_scoped_release>(),
             "Passes the argument to the next node in the pipeline")
        .def("on_error", &ObserverProxy::on_error)
        .def("on_completed", &PyObjectObserver::on_completed, py::call_guard<py::gil_scoped_release>())
        .def_static("make_observer", &ObserverProxy::make_observer);

    py::class_<PyObjectSubscriber>(py_mod, "Subscriber")
        .def("on_next", &SubscriberProxy::on_next, py::call_guard<py::gil_scoped_release>())
        .def("on_error", &SubscriberProxy::on_error)
        .def("on_completed", &PyObjectSubscriber::on_completed, py::call_guard<py::gil_scoped_release>())
        .def("is_subscribed", &SubscriberProxy::is_subscribed, py::call_guard<py::gil_scoped_release>());

    py::class_<PyObjectObservable>(py_mod, "Observable")
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectObserver&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectSubscriber&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("pipe", &ObservableProxy::pipe);

    py_mod.attr("__version__") = MRC_CONCAT_STR(mrc_VERSION_MAJOR << "." << mrc_VERSION_MINOR << "."
                                                                  << mrc_VERSION_PATCH);
}
}  // namespace mrc::pymrc
