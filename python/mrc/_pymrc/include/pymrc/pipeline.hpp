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

#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <string>

namespace mrc::pipeline {
class IPipeline;
}  // namespace mrc::pipeline
namespace mrc::segment {
class IBuilder;
}  // namespace mrc::segment

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

class Pipeline
{
  public:
    Pipeline();
    ~Pipeline();

    /**
     * @brief Create a new MRC segment
     * @param name Segment name
     * @param init initializer used to define segment internals
     */
    void make_segment(const std::string& name, const std::function<void(mrc::segment::IBuilder&)>& init);

    /**
     * @brief
     * @param name Segment name
     * @param ingress_port_info Vector of strings with unique segment ingress port names
     *  note: these must also be unique with respect to egress port ids.
     * @param egress_port_info Vector of strings with unique segment egress port names.
     *  note: these must also be unique with respect to ingress port ids.
     * @param init
     */
    void make_segment(const std::string& name,
                      pybind11::list ingress_port_info,
                      pybind11::list egress_port_info,
                      const std::function<void(mrc::segment::IBuilder&)>& init);

    /**
     * @brief Get the wrapped object held by this Python proxy object
     *
     * @return std::shared_ptr<pipeline::IPipeline>
     */
    std::shared_ptr<pipeline::IPipeline> get_wrapped() const;

  private:
    std::shared_ptr<mrc::pipeline::IPipeline> m_pipeline;
};

#pragma GCC visibility pop
}  // namespace mrc::pymrc
