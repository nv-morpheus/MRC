/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/channel/types.hpp"
#include "mrc/constants.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::node {

template <typename T>
class RxSinkBase : public WritableProvider<T>, public ReadableAcceptor<T>, public SinkChannelOwner<T>, private Watchable
{
  public:
    void sink_add_watcher(std::shared_ptr<WatcherInterface> watcher);
    void sink_remove_watcher(std::shared_ptr<WatcherInterface> watcher);

    void set_timeout(const channel::duration_t& timeout)
    {
        m_timeout = timeout;
    }

  protected:
    RxSinkBase();
    ~RxSinkBase() override = default;

    const rxcpp::observable<T>& observable() const;

  private:
    // this is our channel reader progress engine
    void progress_engine(rxcpp::subscriber<T>& s);

    // observable
    rxcpp::observable<T> m_observable;

    // Start with -max() to indicate an infinite timeout
    channel::duration_t m_timeout{-channel::duration_t::max()};
};

template <typename T>
RxSinkBase<T>::RxSinkBase() :
  m_observable(rxcpp::observable<>::create<T>([this](rxcpp::subscriber<T> s) {
      progress_engine(s);
  }))
{
    // Set the default channel
    this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
}

template <typename T>
const rxcpp::observable<T>& RxSinkBase<T>::observable() const
{
    return m_observable;
}

template <typename T>
void RxSinkBase<T>::progress_engine(rxcpp::subscriber<T>& s)
{
    // Load the context just to help with debugging
    auto& context = mrc::runnable::Context::get_runtime_context();

    T data;

    this->watcher_prologue(WatchableEvent::channel_read, &data);

    channel::Status status;

    while (s.is_subscribed())
    {
        if (m_timeout < channel::duration_t::zero())
        {
            status = this->get_readable_edge()->await_read(data);
        }
        else
        {
            status = this->get_readable_edge()->await_read_for(data, m_timeout);
        }

        if (status == channel::Status::timeout)
        {
            VLOG(10) << context.info() << " await_read_for timed out. Retrying...";
            continue;
        }

        if (status != channel::Status::success)
        {
            break;
        }

        this->watcher_epilogue(WatchableEvent::channel_read, true, &data);
        this->watcher_prologue(WatchableEvent::sink_on_data, &data);

        s.on_next(std::move(data));

        this->watcher_prologue(WatchableEvent::channel_read, &data);
    }

    s.on_completed();
}

template <typename T>
void RxSinkBase<T>::sink_add_watcher(std::shared_ptr<WatcherInterface> watcher)
{
    Watchable::add_watcher(std::move(watcher));
}

template <typename T>
void RxSinkBase<T>::sink_remove_watcher(std::shared_ptr<WatcherInterface> watcher)
{
    Watchable::remove_watcher(std::move(watcher));
}

}  // namespace mrc::node
