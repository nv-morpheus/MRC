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

#include "internal/system/fiber_task_queue.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/core/fiber_pool.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/utils/macros.hpp"
#include "mrc/utils/thread_local_shared_pointer.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace mrc::internal::system {

class FiberPool final : public core::FiberPool
{
  public:
    FiberPool(CpuSet cpu_set, std::vector<std::reference_wrapper<FiberTaskQueue>>&& queues);
    ~FiberPool() final = default;

    DELETE_COPYABILITY(FiberPool);
    DELETE_MOVEABILITY(FiberPool);

    const CpuSet& cpu_set() const final;
    std::size_t thread_count() const final;

    core::FiberTaskQueue& task_queue(const std::size_t& index) final;

    template <typename ResourceT>
    void set_thread_local_resource(std::shared_ptr<ResourceT> resource)
    {
        for (std::uint32_t i = 0; i < thread_count(); ++i)
        {
            set_thread_local_resource(i, resource);
        }
    }

    template <typename ResourceT>
    void set_thread_local_resource(std::uint32_t index, std::shared_ptr<ResourceT> resource)
    {
        task_queue(index)
            .enqueue([resource] { ::mrc::utils::ThreadLocalSharedPointer<ResourceT>::set(resource); })
            .get();
    }

  private:
    CpuSet m_cpu_set;
    std::vector<std::reference_wrapper<FiberTaskQueue>> m_queues;
};

}  // namespace mrc::internal::system
