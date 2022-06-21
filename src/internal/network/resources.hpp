/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/data_plane/server.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/ucx/context.hpp"

#include <srf/cuda/common.hpp>

namespace srf::internal::network {

class Resources final
{
  public:
    Resources(runnable::Resources& runnable_resources);

  private:
    runnable::Resources& m_runnable;
    std::shared_ptr<ucx::Context> m_ucx_context;
    std::unique_ptr<data_plane::Server> m_server;
};

}  // namespace srf::internal::network
