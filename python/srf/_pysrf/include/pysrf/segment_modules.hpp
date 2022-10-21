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

#include "pysrf/types.hpp"

#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/forward.hpp"
#include "srf/segment/object.hpp"

#include <pybind11/functional.h>  // IWYU pragma: keep
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>  // IWYU pragma: keep

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace srf::pysrf {

// Export everything in the srf::pysrf namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class SegmentModuleProxy
{
  public:
    static std::shared_ptr<srf::segment::ObjectProperties> output_port(srf::modules::SegmentModule& self,
                                                                       const std::string& output_id)
    {
        return self.output_port(output_id);
    }

    static std::shared_ptr<srf::segment::ObjectProperties> input_port(srf::modules::SegmentModule& self,
                                                                      const std::string& input_id)
    {
        return self.input_port(input_id);
    }
};

#pragma GCC visibility pop
}  // namespace srf::pysrf