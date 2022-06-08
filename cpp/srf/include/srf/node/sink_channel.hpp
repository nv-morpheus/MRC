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

#include <mutex>
#include <srf/channel/buffered_channel.hpp>
#include <srf/channel/ingress.hpp>
#include <srf/constants.hpp>
#include <srf/exceptions/runtime_error.hpp>
#include <srf/node/edge.hpp>
#include <srf/node/forward.hpp>
#include <srf/node/sink_properties.hpp>
#include <srf/utils/type_utils.hpp>
#include "srf/channel/egress.hpp"

namespace srf::node {

/**
 * @brief Extends SinkProperties to hold a Channel and provide SinkProperties access the the channel ingress
 *
 * @tparam T
 */
template <typename T>
class SinkChannel : public SinkProperties<T>
{
  protected:
    SinkChannel();

  public:
    ~SinkChannel() override = default;

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

    // TODO(#151) - Add property for limiting the number of upstream edges to SourceChannels

  protected:
    /**
     * @brief Enable derived classes to access the reader interface to of the channel
     *
     * @return channel::Egress<T>&
     */
    inline channel::Egress<T>& egress();

  private:
    // implement virtual method from SinkProperties<T>
    [[nodiscard]] std::shared_ptr<channel::Ingress<T>> channel_ingress() final;

    // holds the original channel passed into the constructor or set via update_channel
    std::shared_ptr<Channel<T>> m_channel;

    // holder for the specialized ingress
    std::weak_ptr<Edge<T>> m_ingress;

    // used to make the channel reader persistent by explicitly holding shared_ptr created from calling weak_ptr::lock
    std::shared_ptr<Edge<T>> m_persistent_ingress{nullptr};

    // indicates whether or not a channel connection was ever made
    bool m_ingress_initialized{false};

    // recursive mutex to protect ingress creation; recursion required for persistence
    mutable std::recursive_mutex m_mutex;
};

template <typename T>
SinkChannel<T>::SinkChannel() : m_channel(std::make_unique<channel::BufferedChannel<T>>())
{}

template <typename T>
channel::Egress<T>& SinkChannel<T>::egress()
{
    DCHECK(m_channel);
    return *m_channel;
}

template <typename T>
std::shared_ptr<channel::Ingress<T>> SinkChannel<T>::channel_ingress()
{
    std::shared_ptr<Edge<T>> ingress;
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    CHECK(m_channel);
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
void SinkChannel<T>::update_channel(std::unique_ptr<Channel<T>> channel)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    CHECK(channel);
    CHECK_EQ(m_channel.use_count(), 1) << "can not modify the input channel after it has been shared or is persistent";

    m_channel             = std::move(channel);
    m_ingress_initialized = false;
}

template <typename T>
bool SinkChannel<T>::is_persistent() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return (m_persistent_ingress != nullptr);
}

template <typename T>
void SinkChannel<T>::disable_persistence()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_persistent_ingress = nullptr;
}

template <typename T>
void SinkChannel<T>::enable_persistence()
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    if (m_persistent_ingress)
    {
        return;
    }
    // Get and hold onto the input channel
    auto persistent = channel_ingress();
    CHECK(persistent);
    m_persistent_ingress = std::move(persistent);
}

}  // namespace srf::node
