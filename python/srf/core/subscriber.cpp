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

#include <pybind11/attr.h>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <array>
#include <memory>

namespace srf::pysrf {

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(subscriber, m)
{
    m.doc() = R"pbdoc()pbdoc";

    // Common must be first in every module
    pysrf::import(m, "srf.core.common");

    py::class_<PySubscription>(m, "Subscription");

    py::class_<PyObjectObserver>(m, "Observer")
        .def("on_next",
             &ObserverProxy::on_next,
             py::call_guard<py::gil_scoped_release>(),
             "Passes the argument to the next node in the pipeline")
        .def("on_error", &ObserverProxy::on_error)
        .def("on_completed", &PyObjectObserver::on_completed, py::call_guard<py::gil_scoped_release>())
        .def_static("make_observer", &ObserverProxy::make_observer);

    py::class_<PyObjectSubscriber>(m, "Subscriber")
        .def("on_next", &SubscriberProxy::on_next, py::call_guard<py::gil_scoped_release>())
        .def("on_error", &SubscriberProxy::on_error)
        .def("on_completed", &PyObjectSubscriber::on_completed, py::call_guard<py::gil_scoped_release>())
        .def("is_subscribed", &SubscriberProxy::is_subscribed, py::call_guard<py::gil_scoped_release>());

    py::class_<PyObjectObservable>(m, "Observable")
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectObserver&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("subscribe",
             py::overload_cast<PyObjectObservable*, PyObjectSubscriber&>(&ObservableProxy::subscribe),
             py::call_guard<py::gil_scoped_release>())
        .def("pipe", &ObservableProxy::pipe);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace srf::pysrf
