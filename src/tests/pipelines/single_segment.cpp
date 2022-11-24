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

#include "../segments/common_segments.hpp"
#include "common_pipelines.hpp"

#include "mrc/pipeline/pipeline.hpp"

#include <memory>

// IWYU pragma: no_include "mrc/channel/forward.hpp"

using namespace mrc;

namespace test::pipelines {

std::unique_ptr<pipeline::Pipeline> finite_single_segment()
{
    auto pipeline = pipeline::make_pipeline();
    pipeline->register_segment(test::segments::single_finite_no_ports("seg_1"));
    return pipeline;
}

std::unique_ptr<pipeline::Pipeline> finite_single_segment_will_throw()
{
    auto pipeline = pipeline::make_pipeline();
    pipeline->register_segment(test::segments::single_finite_no_ports_will_throw("seg_1"));
    return pipeline;
}

}  // namespace test::pipelines
