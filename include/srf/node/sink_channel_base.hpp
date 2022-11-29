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

#include "srf/channel/buffered_channel.hpp"
#include "srf/channel/egress.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/constants.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/edge_properties.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/sink_properties.hpp"
#include "srf/utils/type_utils.hpp"

#include <mutex>

namespace srf::node {

/**
 * @brief Extends SinkProperties to hold a Channel and provide SinkProperties access the the channel ingress
 *
 * @tparam T
 */
template <typename T>
class SinkChannelBase
{
  protected:
    SinkChannelBase();

  public:
    /**
     * @brief Enables persistence of the Channel.
     *
     * The default behavior is to close the owned Channel after all Ingress objects created from channel_ingress have
     * been destroyed. The persistent option keeps an internal Ingress object live preventing channel closure.
     *
     * Invoking stop on the Runner which owns the Runnable explicitly disables persistence.
     */
    void enable_persistence();

    /**
     * @brief Disables persistence of the Channel
     */
    void disable_persistence();

    /**
     * @brief Determine if the Channel is persistent or not.
     *
     * @return true
     * @return false
     */
    bool is_persistent() const;

    /**
     * @brief Replace the current Channel.
     *
     * An update can only occur if persistence has not be been requested and no external entities have acquired an
     * Ingress.
     *
     * @param channel
     */
    void update_channel(std::unique_ptr<Channel<T>> channel);

    /**
     * @brief The number of outstanding edge connections from this SinkChannel to SourceChannels
     *
     * @return std::size_t
     */
    std::size_t use_count()
    {
        return m_ingress.use_count();
    }

  protected:
    inline channel::Egress<T>& egress()
    {
        DCHECK(m_channel);
        return *m_channel;
    }

    // access the full channel
    std::shared_ptr<channel::Channel<T>> channel();

    // alternative to update_channel
    // disables ingress_channel
    void set_shared_channel(std::shared_ptr<channel::Channel<T>> channel);

    // thread safe access to
    [[nodiscard]] std::shared_ptr<channel::Ingress<T>> ingress_channel();

  private:
    // holds the original channel passed into the constructor or set via update_channel
    std::shared_ptr<Channel<T>> m_channel;

    // holder for the specialized ingress
    std::weak_ptr<Edge<T>> m_ingress;

    // used to make the channel reader persistent by explicitly holding shared_ptr created from calling weak_ptr::lock
    std::shared_ptr<channel::Ingress<T>> m_persistent_ingress{nullptr};

    // indicates whether or not a channel connection was ever made
    bool m_ingress_initialized{false};

    // indicates if the channel is a unique or shared channel
    // an error will be thrown if attempting to call ingress_channel on a shared channel
    bool m_unique_channel{true};

    // recursive mutex to protect ingress creation; recursion required for persistence
    mutable std::recursive_mutex m_mutex;
};

template <typename T>
SinkChannelBase<T>::SinkChannelBase() : m_channel(std::make_unique<channel::BufferedChannel<T>>())
{}

template <typename T>
std::shared_ptr<channel::Channel<T>> SinkChannelBase<T>::channel()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_channel);
    return m_channel;
}

template <typename T>
std::shared_ptr<channel::Ingress<T>> SinkChannelBase<T>::ingress_channel()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    CHECK(m_channel);
    CHECK(m_unique_channel);
    std::shared_ptr<Edge<T>> ingress;

    if (m_channel->is_channel_closed())
    {
        throw exceptions::SrfRuntimeError("attempting to acquire an Ingress to a closed channel");
    }

    // the last holder that owns a shared_ptr<Ingress> will close the channel
    // when the final instance of the shared_ptr is either reset or goes out of scope
    if ((ingress = m_ingress.lock()))
    {
        return ingress;
    }

    // make a copy of the shared_ptr to the Channel so we can capture it in the deleter
    auto channel_holder = m_channel;

    // create the first shared pointer of this input channel and capture the shared_ptr as part of the deleter
    ingress = std::shared_ptr<Edge<T>>(new Edge<T>(m_channel), [channel_holder](Edge<T>* ptr) {
        channel_holder->close_channel();
        delete ptr;
    });

    // assign the weak_ptr
    m_ingress             = ingress;
    m_ingress_initialized = true;

    return ingress;
}

template <typename T>
void SinkChannelBase<T>::update_channel(std::unique_ptr<Channel<T>> channel)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    CHECK(channel);
    CHECK(!m_ingress_initialized);
    CHECK_EQ(m_channel.use_count(), 1) << "can not modify the input channel after it has been shared or is persistent";
    m_channel = std::move(channel);
}

template <typename T>
void SinkChannelBase<T>::set_shared_channel(std::shared_ptr<Channel<T>> channel)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    CHECK(channel);
    CHECK(m_unique_channel);
    CHECK(!m_ingress_initialized);
    CHECK_EQ(m_channel.use_count(), 1) << "can not modify the input channel after it has been shared or is persistent";

    m_channel        = std::move(channel);
    m_unique_channel = false;
}

template <typename T>
bool SinkChannelBase<T>::is_persistent() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return (m_persistent_ingress != nullptr);
}

template <typename T>
void SinkChannelBase<T>::disable_persistence()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_persistent_ingress = nullptr;
}

template <typename T>
void SinkChannelBase<T>::enable_persistence()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (m_persistent_ingress)
    {
        return;
    }
    // Get and hold onto the input channel
    CHECK(m_persistent_ingress = ingress_channel());
}

}  // namespace srf::node
