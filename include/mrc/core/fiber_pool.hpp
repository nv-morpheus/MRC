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

#include "mrc/core/bitmap.hpp"
#include "mrc/core/task_queue.hpp"
#include "mrc/types.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace mrc::core {

class FiberPool
{
  public:
    virtual ~FiberPool() = default;

    // [[deprecated]] [[nodiscard]] virtual std::shared_ptr<core::FiberTaskQueue> task_queue_shared(
    //     std::size_t index) const = 0;

    [[nodiscard]] virtual const CpuSet& cpu_set() const    = 0;
    [[nodiscard]] virtual std::size_t thread_count() const = 0;

    template <class F, class... ArgsT>
    auto enqueue(std::uint32_t index, F&& f, ArgsT&&... args) -> Future<typename std::result_of<F(ArgsT...)>::type>
    {
        return task_queue(index).enqueue(f, std::forward<ArgsT>(args)...);
    }

    template <class MetaDataT, class F, class... ArgsT>
    auto enqueue(std::uint32_t index, MetaDataT&& md, F&& f, ArgsT&&... args)
        -> Future<typename std::result_of<F(ArgsT...)>::type>
    {
        return task_queue(index).enqueue(std::forward<MetaDataT>(md), std::forward<F>(f), std::forward<ArgsT>(args)...);
    }

  private:
    virtual FiberTaskQueue& task_queue(const std::size_t& index) = 0;
};

}  // namespace mrc::core
