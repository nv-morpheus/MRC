/**
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
#include "mrc/constants.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/edge.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/sink_channel.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::node {

template <typename T, typename ContextT>
class GenericSink : public RxSink<T, ContextT>
{
  public:
    GenericSink();
    ~GenericSink() override = default;

  private:
    virtual void on_data(T&& data) = 0;
};

template <typename T, typename ContextT>
GenericSink<T, ContextT>::GenericSink()
{
    RxSink<T, ContextT>::set_observer(
        rxcpp::make_observer_dynamic<T>([this](T data) { this->on_data(std::move(data)); }));
}

template <typename T>
class GenericSinkComponent : public RxSinkComponent<T>
{
  public:
    GenericSinkComponent()
    {
        RxSinkComponent<T>::set_observer(rxcpp::make_observer_dynamic<T>(
            [this](T data) {
                // Forward to on_data
                this->on_data(std::move(data));
            },
            [this]() {
                // Forward to on_complete
                this->on_complete();
            }));
    }
    ~GenericSinkComponent() override = default;

  private:
    virtual mrc::channel::Status on_data(T&& data) = 0;
    virtual void on_complete()                     = 0;
};

template <typename T>
class LambdaSinkComponent : public GenericSinkComponent<T>
{
  public:
    using on_next_fn_t     = std::function<mrc::channel::Status(T&&)>;
    using on_complete_fn_t = std::function<void()>;

    LambdaSinkComponent(on_next_fn_t on_next_fn) : GenericSinkComponent<T>(), m_on_next_fn(std::move(on_next_fn)) {}

    LambdaSinkComponent(on_next_fn_t on_next_fn, on_complete_fn_t on_complete_fn) :
      GenericSinkComponent<T>(),
      m_on_next_fn(std::move(on_next_fn)),
      m_on_complete_fn(std::move(on_complete_fn))
    {}

    ~LambdaSinkComponent() override = default;

  private:
    channel::Status on_data(T&& t) final
    {
        return m_on_next_fn(std::move(t));
    }

    void on_complete() override
    {
        if (m_on_complete_fn)
        {
            m_on_complete_fn();
        }

        SinkProperties<T>::release_edge_connection();
    }

    on_next_fn_t m_on_next_fn;
    on_complete_fn_t m_on_complete_fn;
};

}  // namespace mrc::node
