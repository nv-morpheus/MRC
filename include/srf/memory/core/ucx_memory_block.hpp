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

#include "srf/memory/core/memory_block.hpp"
//#include "internal/ucx/context.hpp"

#include <glog/logging.h>

namespace srf::memory {

struct ucx_memory_block : public memory_block
{
    //   public:
    //     ucx_memory_block() = default;
    //     ucx_memory_block(void* data, std::size_t bytes) : memory_block(data, bytes) {}
    //     ucx_memory_block(
    //         void* data, std::size_t bytes, ucp_mem_h local_handle, void* remote_handle, std::size_t
    //         remote_handle_size) :
    //       memory_block(data, bytes),
    //       m_local_handle(local_handle),
    //       m_remote_handle(remote_handle),
    //       m_remote_handle_size(remote_handle_size)
    //     {
    //         if (m_remote_handle || m_remote_handle_size)
    //             CHECK(m_remote_handle && m_remote_handle_size);
    //     }
    //     ucx_memory_block(const memory_block& block,
    //                      ucp_mem_h local_handle,
    //                      void* remote_handle,
    //                      std::size_t remote_handle_size) :
    //       memory_block(block),
    //       m_local_handle(local_handle),
    //       m_remote_handle(remote_handle),
    //       m_remote_handle_size(remote_handle_size)
    //     {
    //         if (m_remote_handle || m_remote_handle_size)
    //             CHECK(m_remote_handle && m_remote_handle_size);
    //     }
    //     ~ucx_memory_block() override = default;

    //     ucp_mem_h local_handle() const
    //     {
    //         return m_local_handle;
    //     }
    //     void* remote_handle() const
    //     {
    //         return m_remote_handle;
    //     }
    //     std::size_t remote_handle_size() const
    //     {
    //         return m_remote_handle_size;
    //     }

    //   private:
    //     ucp_mem_h m_local_handle{nullptr};
    //     void* m_remote_handle{nullptr};
    //     std::size_t m_remote_handle_size{0};
};

}  // namespace srf::memory
