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

#include "mrc/modules/mirror_tap/mirror_tap.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/modules/stream_buffer/stream_buffer_base.hpp"

#include <boost/circular_buffer.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <deque>
#include <iostream>
#include <string>

namespace mrc::modules::stream_buffers {

template <typename DataTypeT>
class StreamBufferImmediate : StreamBufferBase<DataTypeT>
{
  public:
    StreamBufferImmediate() : m_ring_buffer_write(128), m_ring_buffer_read(128) {}

    StreamBufferImmediate(std::size_t buffer_size) : m_ring_buffer_write(buffer_size), m_ring_buffer_read(buffer_size)
    {}

    std::size_t buffer_size() override
    {
        return m_ring_buffer_write.capacity();
    }

    void buffer_size(std::size_t size) override
    {
        m_ring_buffer_write.set_capacity(size);
        m_ring_buffer_read.set_capacity(size);
    }

    bool empty() override
    {
        return m_ring_buffer_write.empty();
    }

    void push_back(DataTypeT&& data) override
    {
        std::lock_guard<std::mutex> lock(m_write_mutex);
        m_ring_buffer_write.push_back(std::move(data));
    }

    // Not thread safe for multiple readers
    void flush_next(rxcpp::subscriber<DataTypeT>& subscriber) override
    {
        std::lock_guard<decltype(m_write_mutex)> wlock(m_write_mutex);

        subscriber.on_next(m_ring_buffer_write.front());
        m_ring_buffer_write.pop_front();
    }

    // Not thread safe for multiple readers
    void flush_all(rxcpp::subscriber<DataTypeT>& subscriber) override
    {
        {
            std::lock_guard<decltype(m_write_mutex)> wlock(m_write_mutex);
            // O(1), based on the size of the circular buffer.
            m_ring_buffer_write.swap(m_ring_buffer_read);
        }

        while (!m_ring_buffer_read.empty())
        {
            subscriber.on_next(m_ring_buffer_read.front());
            m_ring_buffer_read.pop_front();
        }
    }

  private:
    std::mutex m_write_mutex;

    boost::circular_buffer<DataTypeT> m_ring_buffer_write;
    boost::circular_buffer<DataTypeT> m_ring_buffer_read;
};
}  // namespace mrc::modules::stream_buffers