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
#include "mrc/node/source_channel.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace mrc::node {

/**
 * @brief Extends SourceChannel<T> to provide observer responsible for writing data to the channel
 *
 * RxSource completes RxSourceBase by providing the observable and subscibable interface.
 *
 * @tparam T
 */
template <typename T>
class RxSourceBase : public SourceChannel<T>, private Watchable
{
  public:
    void source_add_watcher(std::shared_ptr<WatcherInterface> watcher);
    void source_remove_watcher(std::shared_ptr<WatcherInterface> watcher);

  protected:
    RxSourceBase();
    ~RxSourceBase() override = default;

    const rxcpp::observer<T>& observer() const;

  private:
    // the following methods are moved to private from their original scopes to prevent access from deriving classes
    using SourceChannel<T>::await_write;

    rxcpp::observer<T> m_observer;
};

template <typename T>
RxSourceBase<T>::RxSourceBase() :
  m_observer(rxcpp::make_observer_dynamic<T>(
      [this](T data) {
          this->watcher_epilogue(WatchableEvent::sink_on_data, true, &data);
          this->watcher_prologue(WatchableEvent::channel_write, &data);
          SourceChannel<T>::await_write(std::move(data));
          this->watcher_epilogue(WatchableEvent::channel_write, true, &data);
      },
      [](std::exception_ptr ptr) { runnable::Context::get_runtime_context().set_exception(std::move(ptr)); }))
{}

template <typename T>
const rxcpp::observer<T>& RxSourceBase<T>::observer() const
{
    return m_observer;
}

template <typename T>
void RxSourceBase<T>::source_add_watcher(std::shared_ptr<WatcherInterface> watcher)
{
    Watchable::add_watcher(std::move(watcher));
}

template <typename T>
void RxSourceBase<T>::source_remove_watcher(std::shared_ptr<WatcherInterface> watcher)
{
    Watchable::remove_watcher(std::move(watcher));
}

}  // namespace mrc::node
