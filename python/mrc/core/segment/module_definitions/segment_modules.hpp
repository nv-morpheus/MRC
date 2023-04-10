/*
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

#include "mrc/modules/segment_modules.hpp"

#include <pybind11/pybind11.h>  // IWYU pragma: keep

#include <string>

namespace mrc::segment {
class Builder;
}

namespace mrc::pymrc {
#pragma GCC visibility push(default)
// TODO(devin)
class PySegmentModule : public mrc::modules::SegmentModule
{
    using mrc::modules::SegmentModule::SegmentModule;

    void initialize(segment::Builder& builder) override;

    std::string module_type_name() const override;
};

void init_segment_modules(pybind11::module_& module);

#pragma GCC visibility pop
}  // namespace mrc::pymrc
