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

#include "internal/utils/collision_detector.hpp"

#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/engine/segment/forward.hpp"
#include "mrc/types.hpp"

#include <map>
#include <memory>

namespace mrc::internal::pipeline {

class Pipeline
{
  public:
    static std::shared_ptr<Pipeline> unwrap(IPipeline& pipeline);

    void add_segment(std::shared_ptr<const segment::Definition> segment);

    const std::map<SegmentID, std::shared_ptr<const segment::Definition>>& segments() const;

    std::shared_ptr<const segment::Definition> find_segment(SegmentID segment_id) const;

  private:
    utils::CollisionDetector m_segment_hasher;
    utils::CollisionDetector m_port_hasher;

    std::map<SegmentID, std::shared_ptr<const segment::Definition>> m_segments;
};

}  // namespace mrc::internal::pipeline
