/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/node/channel_holder.hpp"

#include <memory>

namespace mrc::node {

template <typename T>
class EdgeChannelReader;

template <typename T>
class EdgeChannelWriter;

template <typename T>
class EdgeChannel;

template <typename T>
class EdgeChannelReader : public IEdgeReadable<T>
{
  public:
    virtual ~EdgeChannelReader()
    {
        if (m_channel)
        {
            if (this->is_connected())
            {
                VLOG(10) << "Closing channel from EdgeChannelReader";
            }

            m_channel->close_channel();
        }
    }

    virtual channel::Status await_read(T& t)
    {
        return m_channel->await_read(t);
    }

  private:
    EdgeChannelReader(std::shared_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}

    std::shared_ptr<mrc::channel::Channel<T>> m_channel;

    template <typename>
    friend class EdgeChannel;
};

template <typename T>
class EdgeChannelWriter : public IEdgeWritable<T>
{
  public:
    virtual ~EdgeChannelWriter()
    {
        if (m_channel)
        {
            if (this->is_connected())
            {
                VLOG(10) << "Closing channel from EdgeChannelWriter";
            }
            m_channel->close_channel();
        }
    }

    virtual channel::Status await_write(T&& t)
    {
        return m_channel->await_write(std::move(t));
    }

  private:
    EdgeChannelWriter(std::shared_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}

    std::shared_ptr<mrc::channel::Channel<T>> m_channel;

    template <typename>
    friend class EdgeChannel;
};

// EdgeChannel holds an actual channel object and provides interfaces for reading/writing
template <typename T>
class EdgeChannel
{
  public:
    EdgeChannel(std::unique_ptr<mrc::channel::Channel<T>> channel) : m_channel(std::move(channel)) {}
    virtual ~EdgeChannel() = default;

    [[nodiscard]] std::shared_ptr<EdgeChannelReader<T>> get_reader() const
    {
        return std::shared_ptr<EdgeChannelReader<T>>(new EdgeChannelReader<T>(m_channel));
    }

    [[nodiscard]] std::shared_ptr<EdgeChannelWriter<T>> get_writer() const
    {
        return std::shared_ptr<EdgeChannelWriter<T>>(new EdgeChannelWriter<T>(m_channel));
    }

  private:
    std::shared_ptr<mrc::channel::Channel<T>> m_channel;
};

}  // namespace mrc::node
