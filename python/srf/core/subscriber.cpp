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

#include "pysrf/subscriber.hpp"

#include "pysrf/types.hpp"  // for PyObjectObserver, PyObjectSubscriber, PyObjectObservable, PySubscription
#include "pysrf/utils.hpp"

#include "srf/version.hpp"

#include <pybind11/attr.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <array>
#include <memory>

namespace srf::pysrf {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(subscriber, module)
{
    module.doc() = R"pbdoc(
        Python bindings for SRF subscribers
        -------------------------------
        .. currentmodule:: subscriber
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // Common must be first in every module
    pysrf::import(module, "srf.core.common");

    py::class_<PySubscription>(module, "Subscription");

    py::class_<PyObjectObserver>(module, "Observer")
        .def("on_next",
             &ObserverProxy::on_next,
             py::call_guard<py::gil_scoped_release>(),
             "Passes the argument to the next node in the pipeline")
        .def("on_error", &ObserverProxy::on_error)
        .def("on_completed", &PyObjectObserver::on_completed, py::call_guard<py::gil_scoped_release>())
        .def_static("make_observer", &ObserverProxy::make_observer);

    py::class_<PyObjectSubscriber>(module, "Subscriber")
        .def("on_next", &SubscriberProxy::on_next, py::call_guard<py::gil_scoped_release>())
        .def("on_error", &SubscriberProxy::on_error)
        .def("on_completed", &PyObjectSubscriber::on_completed, py::call_guard<py::gil_scoped_release>())
        .def("is_subscribed", &SubscriberProxy::is_subscribed, py::call_guard<py::gil_scoped_release>());

    py::class_<PyObjectObservable>(module, "Observable")
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectObserver&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectSubscriber&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("pipe", &ObservableProxy::pipe);

    std::stringstream sstream;
    sstream << srf_VERSION_MAJOR << "." << srf_VERSION_MINOR << "." << srf_VERSION_PATCH;

    module.attr("__version__") = sstream.str();
}
}  // namespace srf::pysrf
