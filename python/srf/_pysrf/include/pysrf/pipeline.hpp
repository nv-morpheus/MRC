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

#include <srf/pipeline/pipeline.hpp>
#include <srf/segment/builder.hpp>

#include <functional>
#include <memory>
#include <string>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class Pipeline
{
  public:
    Pipeline();

    void make_segment(const std::string& name, const std::function<void(srf::segment::Builder&)>& init);

    std::unique_ptr<srf::pipeline::Pipeline> swap();

  private:
    std::unique_ptr<srf::pipeline::Pipeline> m_pipeline;
};

#pragma GCC visibility pop
}  // namespace srf::pysrf
