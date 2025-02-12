/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Forward pybind11 decs
namespace pybind11 {
class module_;
class object;
}  // namespace pybind11

// Common MRC classes that are used
namespace mrc::segment {

class ObjectProperties;

}  // namespace mrc::segment

namespace mrc::pymrc {

class PyObjectHolder;
class PyObjectWrapper;
class PythonOperator;

}  // namespace mrc::pymrc
