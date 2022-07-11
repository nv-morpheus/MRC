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
#include "srf/channel/channel.hpp"
#include "srf/constants.hpp"
#include "srf/core/utils.hpp"
#include "srf/core/watcher.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/rx_epilogue_tap.hpp"
#include "srf/node/rx_prologue_tap.hpp"
#include "srf/node/rx_sink_base.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/node/rx_subscribable.hpp"
#include "srf/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx-subscription.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace srf::node {

template <typename InputT, typename OutputT, typename ContextT>
class RxNode : public RxSinkBase<InputT>,
               public RxSourceBase<OutputT>,
               public RxRunnable<ContextT>,
               public RxPrologueTap<InputT>,
               public RxEpilogueTap<OutputT>
{
  public:
    // function defining the stream, i.e. operations linking Sink -> Source
    using stream_fn_t = std::function<rxcpp::observable<OutputT>(const rxcpp::observable<InputT>&)>;

    RxNode();

    template <typename... OpsT>
    RxNode(OpsT&&... ops);

    ~RxNode() override = default;

    template <typename... OpsT>
    RxNode& pipe(OpsT&&... ops)
    {
        make_stream([=](auto start) { return (start | ... | ops); });
        return *this;
    }

    void make_stream(stream_fn_t fn);

  private:
    // the following method(s) are moved to private from their original scopes to prevent access from deriving classes
    using RxSinkBase<InputT>::observable;
    using RxSourceBase<OutputT>::observer;

    void do_subscribe(rxcpp::composite_subscription& subscription) final;
    void on_shutdown_critical_section() final;

    void on_stop(const rxcpp::subscription& subscription) override;
    void on_kill(const rxcpp::subscription& subscription) final;

    // m_stream works like an operator. It is a function taking an observable and returning an observable. Allows
    // delayed construction of the observable chain for prologue/epilogue
    stream_fn_t m_stream;
};

template <typename InputT, typename OutputT, typename ContextT>
RxNode<InputT, OutputT, ContextT>::RxNode() :
  m_stream([](const rxcpp::observable<InputT>& obs) {
      // Default to just returning the input
      return obs;
  })
{}

template <typename InputT, typename OutputT, typename ContextT>
template <typename... OpsT>
RxNode<InputT, OutputT, ContextT>::RxNode(OpsT&&... ops)
{
    pipe(std::forward<OpsT>(ops)...);
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::make_stream(stream_fn_t fn)
{
    m_stream = std::move(fn);
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::do_subscribe(rxcpp::composite_subscription& subscription)
{
    // Start with the base sinke observable
    auto observable_in = RxSinkBase<InputT>::observable();

    // Apply prologue taps
    observable_in = this->apply_prologue_taps(observable_in);

    // Apply the specified stream
    auto observable_out = m_stream(observable_in);

    // Apply epilogue taps
    observable_out = this->apply_epilogue_taps(observable_out);

    // Subscribe to the observer
    observable_out.subscribe(subscription, RxSourceBase<OutputT>::observer());
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::on_stop(const rxcpp::subscription& subscription)
{
    this->disable_persistence();
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::on_kill(const rxcpp::subscription& subscription)
{
    this->disable_persistence();
    subscription.unsubscribe();
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::on_shutdown_critical_section()
{
    DVLOG(10) << runnable::Context::get_runtime_context().info() << " releasing source channel";
    RxSourceBase<OutputT>::release_channel();
}

}  // namespace srf::node
