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

#include <memory>

namespace mrc::segment {
class IDefinition;
class Definition;
}  // namespace mrc::segment

namespace mrc::pipeline {

class PipelineDefinition;

class IPipelineBase
{
  public:
    IPipelineBase();
    virtual ~IPipelineBase() = 0;

  protected:
    void register_segment(std::shared_ptr<const segment::IDefinition> segment);

  private:
    void add_segment(std::shared_ptr<const segment::Definition> segment);

    std::shared_ptr<PipelineDefinition> m_impl;
    friend PipelineDefinition;
};

}  // namespace mrc::pipeline
