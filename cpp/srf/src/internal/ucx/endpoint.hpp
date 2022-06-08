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

#include <srf/types.hpp>
#include "internal/ucx/common.hpp"
#include "internal/ucx/primitive.hpp"

#include <ucp/api/ucp_def.h>  // for ucp_ep_h

namespace srf::internal::ucx {

class Endpoint : public Primitive<ucp_ep_h>
{
  public:
    Endpoint(Handle<Worker>, WorkerAddress);
    ~Endpoint() override;

  private:
    Handle<Worker> m_worker;
};

}  // namespace srf::internal::ucx
