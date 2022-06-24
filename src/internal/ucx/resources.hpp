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

#include "internal/resources/runnable_provider.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/worker.hpp"

namespace srf::internal::ucx {

class Resources final : public resources::PartitionResourceBase
{
  public:
    Resources(resources::PartitionResourceBase& partition_provider);

    Context& context();

  private:
    std::shared_ptr<Context> m_ucx_context;
    std::shared_ptr<Worker> m_worker_server;
    std::shared_ptr<Worker> m_worker_client;
};

}  // namespace srf::internal::ucx
