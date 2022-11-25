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

#include "mrc/channel/status.hpp"
#include "mrc/data/reusable_pool.hpp"
#include "mrc/utils/macros.hpp"

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <memory>

using namespace mrc;

namespace {
class Buffer
{
  public:
    Buffer() = default;

    DELETE_COPYABILITY(Buffer);

    float* data()
    {
        return m_buffer.data();
    }

    const float* data() const
    {
        return m_buffer.data();
    }

    std::size_t size() const
    {
        return m_buffer.size();
    }

  private:
    std::array<float, 1024> m_buffer;
};
}  // namespace

static void mrc_data_reusable(benchmark::State& state)
{
    auto pool = data::ReusablePool<Buffer>::create(32);
    pool->add_item(std::make_unique<Buffer>());
    pool->add_item(std::make_unique<Buffer>());

    for (auto _ : state)
    {
        auto buffer = pool->await_item();
        benchmark::DoNotOptimize(buffer->data()[0] += 1.0);
    }
}

BENCHMARK(mrc_data_reusable);
