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

#include "mrc/data/reusable_pool.hpp"
#include "mrc/memory/literals.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/utils/macros.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <utility>

namespace test::nodes {

using namespace mrc::memory::literals;

template <std::size_t Bytes>
class Buffer
{
  public:
    Buffer() = default;

    DELETE_COPYABILITY(Buffer);

    void* data()
    {
        return m_buffer.data();
    }

    const void* data() const
    {
        return m_buffer.data();
    }

    std::size_t size() const
    {
        return m_buffer.size();
    }

  private:
    std::array<char, Bytes> m_buffer;
};

template <std::size_t Bytes>
class BufferSource final : public mrc::node::GenericSource<mrc::data::Reusable<Buffer<Bytes>>>
{
  public:
    using data_t = mrc::data::Reusable<Buffer<Bytes>>;

    BufferSource(std::size_t capacity) : m_pool(mrc::data::ReusablePool<Buffer<Bytes>>::create(capacity)) {}

    mrc::data::ReusablePool<Buffer<Bytes>>& pool()
    {
        CHECK(m_pool);
        return *m_pool;
    }

  private:
    void data_source(rxcpp::subscriber<data_t>& s) final
    {
        auto buffer = m_pool->await_item();
        // populate buffer
        s.on_next(std::move(buffer));
    }

    std::shared_ptr<mrc::data::ReusablePool<Buffer<Bytes>>> m_pool;
};

std::unique_ptr<mrc::node::RxSource<int>> finite_int_rx_source(int count = 3);
std::unique_ptr<mrc::node::RxSource<int>> infinite_int_rx_source();

std::unique_ptr<mrc::node::RxSink<int>> int_sink();
std::unique_ptr<mrc::node::RxSink<int>> int_sink_throw_on_even();

std::unique_ptr<BufferSource<1_MiB>> infinte_buffer_source(std::size_t capacity);

}  // namespace test::nodes
