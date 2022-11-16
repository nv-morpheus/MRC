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

#include "srf/modules/segment_modules.hpp"
#include "srf/segment/forward.hpp"

#include <nlohmann/json.hpp>
#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/stl.h>         // IWYU pragma: keep

#include <functional>
#include <string>

namespace srf::pysrf {

class ModuleRegistryProxy;

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

/**
 * PythonSegmentModule exists to solve one problem: allowing for binding a dynamic initializer for a SegmentModule
 * This is accomplished by allowing the builder to set m_py_initialize, and subsequently calling it in the overridden
 * `initialize` method.
 */
class PythonSegmentModule : public srf::modules::SegmentModule
{
    using type_t = PythonSegmentModule;
    friend ModuleRegistryProxy;

  public:
    using py_initializer_t = std::function<void(srf::segment::Builder&)>;

    PythonSegmentModule(std::string module_name);
    PythonSegmentModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;
    std::string module_type_name() const override;

  private:
    py_initializer_t m_py_initialize{};
};
}  // namespace srf::pysrf

#pragma GCC visibility pop