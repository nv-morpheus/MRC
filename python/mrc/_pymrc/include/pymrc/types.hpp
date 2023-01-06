/**
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

#pragma once

#include "pymrc/utilities/object_wrappers.hpp"  // IWYU Pragma: export

#include "mrc/segment/object.hpp"

#include <rxcpp/rx.hpp>

#include <functional>

namespace mrc::pymrc {

// NOLINTBEGIN(readability-identifier-naming)
using PyHolder           = PyObjectHolder;
using PySubscription     = rxcpp::subscription;
using PyObjectObserver   = rxcpp::observer<PyHolder, void, void, void, void>;
using PyObjectSubscriber = rxcpp::subscriber<PyHolder, PyObjectObserver>;
using PyObjectObservable = rxcpp::observable<PyHolder>;
using PyNode             = mrc::segment::ObjectProperties;
using PyObjectOperateFn  = std::function<PyObjectObservable(PyObjectObservable source)>;
// NOLINTEND(readability-identifier-naming)

}  // namespace mrc::pymrc
