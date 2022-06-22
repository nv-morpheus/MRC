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

#include "internal/runnable/engine.hpp"

#include <srf/constants.hpp>
#include <srf/core/fiber_meta_data.hpp>
#include <srf/core/task_queue.hpp>
#include <srf/runnable/types.hpp>
#include <srf/types.hpp>

#include <functional>
#include <memory>

namespace srf::internal::runnable {

class FiberEngine final : public Engine
{
  public:
    FiberEngine(std::shared_ptr<core::FiberTaskQueue> task_queue, int priority = SRF_DEFAULT_FIBER_PRIORITY);
    FiberEngine(std::shared_ptr<core::FiberTaskQueue> task_queue, const FiberMetaData& meta);

    ~FiberEngine() final = default;

    EngineType engine_type() const final;

  private:
    Future<void> do_launch_task(std::function<void()> task) final;

    std::shared_ptr<core::FiberTaskQueue> m_task_queue;
    FiberMetaData m_meta;
};

}  // namespace srf::internal::runnable
