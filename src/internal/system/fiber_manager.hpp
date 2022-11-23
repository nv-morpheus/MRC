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

#include "internal/system/fiber_pool.hpp"
#include "internal/system/fiber_task_queue.hpp"

#include "mrc/core/bitmap.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>

#include <cstdint>
#include <map>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>
#include <vector>

namespace mrc::internal::system {

class Resources;  // IWYU pragma: keep

class FiberManager final
{
  public:
    FiberManager(const Resources& resources);
    ~FiberManager();

    DELETE_COPYABILITY(FiberManager);
    DELETE_MOVEABILITY(FiberManager)

    [[nodiscard]] FiberTaskQueue& task_queue(std::uint32_t cpu_id) const;
    [[nodiscard]] FiberPool make_pool(CpuSet cpu_set) const;

    template <class F>
    [[nodiscard]] auto enqueue_fiber(std::uint32_t queue_idx, const F& to_enqueue) const
        -> Future<typename std::result_of<F(std::uint32_t)>::type>
    {
        using return_vec_t = Future<typename std::result_of<F(std::uint32_t)>::type>;
        return_vec_t future;

        auto found = m_queues.find(queue_idx);

        CHECK(found != m_queues.end()) << "Index is not in list of queues. Queue Index: " << queue_idx;

        CHECK(found->second);
        future = found->second->enqueue([to_enqueue, queue_idx]() {
            // Call the user supplied function
            return to_enqueue(queue_idx);
        });

        return future;
    }

    // Runs a function on each thread in this manager. Each function will receive the thread index as the only argument
    template <class F>
    [[nodiscard]] auto enqueue_fiber_on_all(const F& to_enqueue) const
        -> std::vector<Future<typename std::result_of<F(std::uint32_t)>::type>>
    {
        return enqueue_fiber_on_cpuset(m_cpu_set, std::move(to_enqueue));
    }

    template <class F>
    [[nodiscard]] auto enqueue_fiber_on_cpuset(const CpuSet& cpu_set, const F& to_enqueue) const
        -> std::vector<Future<typename std::result_of<F(std::uint32_t)>::type>>
    {
        using return_vec_t = std::vector<Future<typename std::result_of<F(std::uint32_t)>::type>>;
        return_vec_t futures;

        CHECK(m_cpu_set.contains(cpu_set));

        for (const auto& x : m_queues)
        {
            std::uint32_t idx = x.first;
            if (cpu_set.is_set(idx))
            {
                CHECK(x.second);
                futures.push_back(x.second->enqueue([to_enqueue, idx]() {
                    // Call the user supplied function
                    return to_enqueue(idx);
                }));
            }
        }

        return futures;
    }

  private:
    void stop();
    void join();

    const CpuSet m_cpu_set;
    std::map<std::uint32_t, std::unique_ptr<FiberTaskQueue>> m_queues;
};

}  // namespace mrc::internal::system
