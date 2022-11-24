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

#pragma once

#include "pymrc/types.hpp"

#include <pybind11/pytypes.h>  // for pybind11::object

#include <functional>

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class ObserverProxy
{
  public:
    static void on_next(PyObjectObserver* self, pybind11::object&& value);
    static void on_error(PyObjectObserver* self, pybind11::object&& value);

    static PyObjectObserver make_observer(std::function<void(pybind11::object x)> on_next,
                                          std::function<void(pybind11::object x)> on_error,
                                          std::function<void()> on_completed);
};

class SubscriberProxy
{
  public:
    static void on_next(PyObjectSubscriber* self, pybind11::object&& value);
    static void on_error(PyObjectSubscriber* self, pybind11::object&& value);
    static bool is_subscribed(PyObjectSubscriber* self);
};

class ObservableProxy
{
  public:
    static PySubscription subscribe(PyObjectObservable* self, PyObjectObserver& observer);
    static PySubscription subscribe(PyObjectObservable* self, PyObjectSubscriber& subscriber);
    static PyObjectObservable pipe(PyObjectObservable* self, pybind11::args args);
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
