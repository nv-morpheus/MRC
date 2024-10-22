/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/constants.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/rx_epilogue_tap.hpp"
#include "mrc/node/rx_prologue_tap.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_source_base.hpp"
#include "mrc/node/rx_subscribable.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace mrc::node {

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
        make_stream([=](auto start) {
            return (start | ... | ops);
        });
        return *this;
    }

    void make_stream(stream_fn_t fn);

    void on_shutdown_critical_section() final;

  private:
    // the following method(s) are moved to private from their original scopes to prevent access from deriving classes
    using RxSinkBase<InputT>::observable;
    using RxSourceBase<OutputT>::observer;

    void do_subscribe(rxcpp::composite_subscription& subscription) final;

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
    // this->disable_persistence();
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::on_kill(const rxcpp::subscription& subscription)
{
    // this->disable_persistence();
    subscription.unsubscribe();
}

template <typename InputT, typename OutputT, typename ContextT>
void RxNode<InputT, OutputT, ContextT>::on_shutdown_critical_section()
{
    DVLOG(10) << runnable::Context::get_runtime_context().info() << " releasing source channel";
    RxSourceBase<OutputT>::release_edge_connection();
}

template <typename T>
class EdgeRxSubscriber : public edge::IEdgeWritable<T>
{
  public:
    using subscriber_t = rxcpp::subscriber<T>;

    EdgeRxSubscriber(subscriber_t subscriber) : m_subscriber(subscriber) {}

    ~EdgeRxSubscriber()
    {
        if (this->is_connected())
        {
            m_subscriber.on_completed();
        }
    }

    void set_subscriber(subscriber_t subscriber)
    {
        m_subscriber = subscriber;
    }

    virtual channel::Status await_write(T&& t)
    {
        m_subscriber.on_next(std::move(t));

        return channel::Status::success;
    }

  private:
    subscriber_t m_subscriber;
};

template <typename InputT, typename OutputT>
class RxNodeComponent : public WritableProvider<InputT>, public WritableAcceptor<OutputT>
{
  public:
    using stream_fn_t = std::function<rxcpp::observable<OutputT>(const rxcpp::observable<InputT>&)>;

    RxNodeComponent()
    {
        auto edge = std::make_shared<EdgeRxSubscriber<InputT>>(m_subject.get_subscriber());

        WritableProvider<InputT>::init_owned_edge(edge);
    }

    RxNodeComponent(stream_fn_t stream_fn) : RxNodeComponent()
    {
        this->make_stream(stream_fn);
    }

    template <typename... OpsT>
    RxNodeComponent(OpsT&&... ops) : RxNodeComponent()
    {
        this->pipe(std::forward<OpsT>(ops)...);
    }

    template <typename... OpsT>
    RxNodeComponent& pipe(OpsT&&... ops)
    {
        make_stream([=](auto start) {
            return (start | ... | ops);
        });
        return *this;
    }

    void make_stream(stream_fn_t fn)
    {
        if (m_subject_subscription.is_subscribed())
        {
            m_subject_subscription.unsubscribe();
        }

        // Start with the base sinke observable
        auto observable_in = m_subject.get_observable();

        // // Apply prologue taps
        // observable_in = this->apply_prologue_taps(observable_in);

        // Apply the specified stream
        auto observable_out = fn(observable_in);

        // // Apply epilogue taps
        // observable_out = this->apply_epilogue_taps(observable_out);

        // Subscribe to the observer
        m_subject_subscription = observable_out.subscribe(rxcpp::make_observer_dynamic<OutputT>(
            [this](OutputT message) {
                // Forward to the writable edge
                this->get_writable_edge()->await_write(std::move(message));
            },
            [this](std::exception_ptr ptr) {
                WritableAcceptor<OutputT>::release_edge_connection();
                runnable::Context::get_runtime_context().set_exception(std::move(ptr));
            },
            [this]() {
                // On completion, release connections
                WritableAcceptor<OutputT>::release_edge_connection();
            }));
    }

  private:
    rxcpp::subjects::subject<InputT> m_subject;
    rxcpp::subscription m_subject_subscription;
};

}  // namespace mrc::node
