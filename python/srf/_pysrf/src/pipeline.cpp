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

#include <pysrf/pipeline.hpp>

#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>  // for move

namespace srf::pysrf {

namespace py = pybind11;

Pipeline::Pipeline() : m_pipeline(srf::pipeline::make_pipeline()) {}

void Pipeline::make_segment(const std::string& name, const std::function<void(srf::segment::Builder&)>& init)
{
    auto init_wrapper = [=](srf::segment::Builder& seg) {
        py::gil_scoped_acquire gil;
        init(seg);
    };

    m_pipeline->make_segment(name, init_wrapper);
}

std::unique_ptr<srf::pipeline::Pipeline> Pipeline::swap()
{
    auto tmp   = std::move(m_pipeline);
    m_pipeline = srf::pipeline::make_pipeline();
    return std::move(tmp);
}
}  // namespace srf::pysrf
