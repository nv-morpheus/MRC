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

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/object.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <string>

namespace srf::pysrf {

namespace py = pybind11;

class ModuleRegistryProxy;

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class PythonSegmentModule : public srf::modules::SegmentModule
{
    friend ModuleRegistryProxy;
  public:
    using py_initializer_t = std::function<void(segment::Builder&)>;

    PythonSegmentModule(std::string module_name);
    PythonSegmentModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    py_initializer_t m_py_initialize;
};

PythonSegmentModule::PythonSegmentModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

PythonSegmentModule::PythonSegmentModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void PythonSegmentModule::initialize(segment::Builder& builder)
{
    VLOG(2) << "Calling PythonSegmentModule::initialize";
    m_py_initialize(std::forward<segment::Builder&>(builder));
    VLOG(2) << "Calling PythonSegmentModule::initialize -> DONE";
}

}  // namespace srf::pysrf

#pragma GCC visibility pop