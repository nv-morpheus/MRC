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

#include "internal/ucx/common.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/endpoint.hpp"
#include "internal/ucx/primitive.hpp"

#include "srf/types.hpp"

#include <ucp/api/ucp_def.h>  // for ucp_worker_h, ucp_address_t

#include <cstddef>  // for size_t
#include <string>

namespace srf::internal::ucx {

class Worker : public Primitive<ucp_worker_h>
{
  public:
    Worker(Handle<Context>);
    ~Worker() override;

    unsigned progress();

    const std::string& address();
    void release_address();

    Handle<Endpoint> create_endpoint(WorkerAddress);

    Context& context();

  private:
    Handle<Context> m_context;
    std::string m_address;
    ucp_address_t* m_address_pointer;
    std::size_t m_address_length;
};

}  // namespace srf::internal::ucx
