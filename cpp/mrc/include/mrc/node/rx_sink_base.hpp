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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/constants.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/edge.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/sink_channel.hpp"
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
class RxSinkBase : public SinkChannel<T>, private Watchable
{
  public:
    void sink_add_watcher(std::shared_ptr<WatcherInterface> watcher);
    void sink_remove_watcher(std::shared_ptr<WatcherInterface> watcher);

  protected:
    RxSinkBase();
    ~RxSinkBase() override = default;

    const rxcpp::observable<T>& observable() const;

  private:
    // the following methods are moved to private from their original scopes to prevent access from deriving classes
    using SinkChannel<T>::egress;

    // this is our channel reader progress engine
    void progress_engine(rxcpp::subscriber<T>& s);

    // observable
    rxcpp::observable<T> m_observable;
};

template <typename T>
RxSinkBase<T>::RxSinkBase() :
  SinkChannel<T>(),
  m_observable(rxcpp::observable<>::create<T>([this](rxcpp::subscriber<T> s) { progress_engine(s); }))
{}

template <typename T>
const rxcpp::observable<T>& RxSinkBase<T>::observable() const
{
    return m_observable;
}

template <typename T>
void RxSinkBase<T>::progress_engine(rxcpp::subscriber<T>& s)
{
    T data;
    this->watcher_prologue(WatchableEvent::channel_read, &data);
    while (s.is_subscribed() && (SinkChannel<T>::egress().await_read(data) == channel::Status::success))
    {
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
