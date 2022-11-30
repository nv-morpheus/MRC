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

#include "internal/pipeline/instance.hpp"
#include "internal/pipeline/types.hpp"

#include "mrc/node/generic_sink.hpp"

#include <memory>
#include <string>

namespace mrc::internal::pipeline {

class Controller final : public node::GenericSink<ControlMessage>
{
  public:
    Controller(std::unique_ptr<Instance> pipeline);

    void await_on_pipeline() const;

  private:
    void on_data(ControlMessage&& message) final;
    void did_complete() final;

    void update(SegmentAddresses&& new_segments);
    void stop();
    void kill();

    static const std::string& info();

    std::unique_ptr<Instance> m_pipeline;
    SegmentAddresses m_current_segments;
};

}  // namespace mrc::internal::pipeline
