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

#include "srf/experimental/modules/segment_modules.hpp"

namespace srf::modules {

SegmentModule::SegmentModule(std::string module_name) : m_module_name(std::move(module_name))
{
    std::stringstream sstream;

    sstream << "segment_module/" << m_module_name << "/";
    m_component_prefix = sstream.str();
}

const std::string& SegmentModule::name() const
{
    return m_module_name;
}

const std::string& SegmentModule::component_prefix() const
{
    return m_component_prefix;
}

}  // namespace srf::modules