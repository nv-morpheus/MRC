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

#include "pysrf/py_segment_module.hpp"

#include "srf/core/utils.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace srf::pysrf {

PythonSegmentModule::PythonSegmentModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

PythonSegmentModule::PythonSegmentModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void PythonSegmentModule::initialize(segment::Builder& builder)
{
    VLOG(2) << "Calling PythonSegmentModule::initialize";
    m_py_initialize(builder);
}

std::string PythonSegmentModule::module_type_name() const
{
    return std::string(::srf::type_name<type_t>());
}

}  // namespace srf::pysrf