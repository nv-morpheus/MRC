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

#include "internal/ucx/primitive.hpp"

#include <ucp/api/ucp_def.h>  // for ucp_mem_h, ucp_context_h

#include <cstddef>  // for size_t
#include <tuple>

namespace mrc::internal::ucx {

class Context final : public Primitive<ucp_context_h>
{
  public:
    Context();
    ~Context() override;

    ucp_mem_h register_memory(const void*, std::size_t);

    std::tuple<ucp_mem_h, void*, std::size_t> register_memory_with_rkey(const void*, std::size_t);

    void unregister_memory(ucp_mem_h, void* rbuffer = nullptr);
};

}  // namespace mrc::internal::ucx
