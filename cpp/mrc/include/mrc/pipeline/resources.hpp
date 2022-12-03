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

#include "mrc/core/fiber_meta_data.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/metrics/registry.hpp"
#include "mrc/runnable/launch_control.hpp"

namespace mrc::pipeline {

struct Resources
{
    virtual ~Resources() = default;

    virtual core::FiberTaskQueue& main()              = 0;
    virtual runnable::LaunchControl& launch_control() = 0;
    // virtual std::shared_ptr<metrics::Registry> metrics_registry() = 0;
};

}  // namespace mrc::pipeline
