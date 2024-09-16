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

#include "pymrc/operators.hpp"
#include "pymrc/types.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/eval.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mrc::pymrc {

namespace py = pybind11;
using namespace py::literals;

void ObserverProxy::on_next(PyObjectObserver* self, py::object&& value)
{
    self->on_next(std::move(value));
}

void ObserverProxy::on_error(PyObjectObserver* self, py::object&& value)
{
    try
    {
        //  Raise the exception locally to convert to C++ exception
        auto locals = py::dict("__on_error_exception"_a = std::move(value));
        py::exec("raise __on_error_exception", py::globals(), locals);
    } catch (py::error_already_set)
    {
        py::gil_scoped_release nogil;

        self->on_error(std::current_exception());
    }
}

PyObjectObserver ObserverProxy::make_observer(std::function<void(py::object x)> on_next,
                                              std::function<void(py::object x)> on_error,
                                              std::function<void()> on_completed)
{
    CHECK(on_next);
    CHECK(on_error);
    CHECK(on_completed);

    auto on_next_w = [on_next](PyHolder x) {
        pybind11::gil_scoped_acquire gil;
        on_next(std::move(x));  // Move the object into a temporary
    };

    auto on_error_w = [on_error](std::exception_ptr x) {
        pybind11::gil_scoped_acquire gil;
        on_error(py::none());
    };

    auto on_completed_w = [on_completed]() {
        pybind11::gil_scoped_acquire gil;
        on_completed();
    };

    return rxcpp::make_observer_dynamic<PyHolder>(on_next_w, on_error_w, on_completed_w);
}

void SubscriberProxy::on_next(PyObjectSubscriber* self, py::object&& value)
{
    // Check to see if we are subscribed before sending the value
    if (self->is_subscribed())
    {
        self->on_next(std::move(value));
    }
};

void SubscriberProxy::on_error(PyObjectSubscriber* self, py::object&& value)
{
    try
    {
        //  Raise the exception locally to convert to C++ exception
        auto locals = py::dict("__on_error_exception"_a = std::move(value));
        py::exec("raise __on_error_exception", py::globals(), locals);
    } catch (py::error_already_set)
    {
        py::gil_scoped_release nogil;

        self->on_error(std::current_exception());
    }
};

bool SubscriberProxy::is_subscribed(PyObjectSubscriber* self)
{
    // No GIL here
    return self->is_subscribed();
}

bool SubscriptionProxy::is_subscribed(PySubscription* self)
{
    // No GIL here
    return self->is_subscribed();
}

PySubscription ObservableProxy::subscribe(PyObjectObservable* self, PyObjectObserver& observer)
{
    // Call the internal subscribe function
    return self->subscribe(observer);
}

PySubscription ObservableProxy::subscribe(PyObjectObservable* self, PyObjectSubscriber& subscriber)
{
    // Call the internal subscribe function
    return self->subscribe(subscriber);
}

PyObjectObservable ObservableProxy::pipe(const PyObjectObservable* self, py::args args)
{
    std::vector<PyObjectOperateFn> operators;

    for (const auto& a : args)
    {
        if (!py::isinstance<PythonOperator>(a))
        {
            throw std::runtime_error("All arguments must be Operators");
        }

        auto op = a.cast<PythonOperator>();

        operators.emplace_back(op.get_operate_fn());
    }

    if (operators.empty())
    {
        throw std::runtime_error("pipe() must be given at least one argument");
    }

    auto result = *self | operators[0];

    for (auto i = 1; i < operators.size(); i++)
    {
        result = result | operators[i];
    }

    return result;
}

}  // namespace mrc::pymrc
