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
#include "mrc/constants.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/core/watcher.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/rx_prologue_tap.hpp"
#include "mrc/node/rx_runnable.hpp"
#include "mrc/node/rx_sink_base.hpp"
#include "mrc/node/rx_subscribable.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::node {

template <typename T>
class EdgeRxObserver : public edge::IEdgeWritable<T>
{
  public:
    using observer_t = rxcpp::observer<T>;

    EdgeRxObserver() = default;

    ~EdgeRxObserver()
    {
        if (this->is_connected())
        {
            m_observer.on_completed();
        }
    }

    void set_observer(observer_t observer)
    {
        m_observer = observer;
    }

    virtual channel::Status await_write(T&& t)
    {
        m_observer.on_next(std::move(t));

        return channel::Status::success;
    }

  private:
    observer_t m_observer;
};

template <typename T, typename ContextT>
class RxSink : public RxSinkBase<T>, public RxRunnable<ContextT>, public RxPrologueTap<T>
{
  public:
    using observer_t       = rxcpp::observer<T>;
    using on_next_fn_t     = std::function<void(T)>;
    using on_error_fn_t    = std::function<void(std::exception_ptr)>;
    using on_complete_fn_t = std::function<void()>;

    RxSink(std::string name = std::string()) : RxSinkBase<T>(name), SinkProperties<T>(name)
    {
        LOG(INFO) << "RxSink constructor";
    }

    ~RxSink() override = default;

    template <typename... ArgsT>
    RxSink(ArgsT&&... args)
    {
        set_observer(std::forward<ArgsT>(args)...);
    }

    template <typename... ArgsT>
    RxSink(std::string name, ArgsT&&... args) : RxSinkBase<T>(name), SinkProperties<T>(name)
    {
        set_observer(std::forward<ArgsT>(args)...);
    }

    void set_observer(observer_t observer);

    template <typename... ArgsT>
    void set_observer(ArgsT&&... args)
    {
        set_observer(rxcpp::make_observer_dynamic<T>(std::forward<ArgsT>(args)...));
    }

  private:
    // the following methods are moved to private from their original scopes to prevent access from deriving classes
    using RxSinkBase<T>::observable;

    void on_shutdown_critical_section() final;
    void do_subscribe(rxcpp::composite_subscription& subscription) final;

    void on_stop(const rxcpp::subscription& subscription) final;
    void on_kill(const rxcpp::subscription& subscription) final;

    observer_t m_observer;
};

template <typename T, typename ContextT>
void RxSink<T, ContextT>::set_observer(rxcpp::observer<T> observer)
{
    m_observer = std::move(observer);
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::do_subscribe(rxcpp::composite_subscription& subscription)
{
    auto observable = RxPrologueTap<T>::apply_prologue_taps(RxSinkBase<T>::observable());

    auto default_error_handler = rxcpp::make_observer_dynamic<T>(
        [this](T data) {
            m_observer.on_next(std::move(data));
        },
        [this](std::exception_ptr ptr) {
            runnable::Context::get_runtime_context().set_exception(std::move(std::current_exception()));
            try
            {
                m_observer.on_error(std::move(ptr));
            } catch (...)
            {
                runnable::Context::get_runtime_context().set_exception(std::move(std::current_exception()));
            }
        },
        [this] {
            m_observer.on_completed();
        });

    observable.subscribe(subscription, default_error_handler);
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_stop(const rxcpp::subscription& subscription)
{
    // this->disable_persistence();
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_kill(const rxcpp::subscription& subscription)
{
    // this->disable_persistence();
    subscription.unsubscribe();
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_shutdown_critical_section()
{}

template <typename T>
class RxSinkComponent : public WritableProvider<T>
{
  public:
    using observer_t       = rxcpp::observer<T>;
    using on_next_fn_t     = std::function<void(T)>;
    using on_error_fn_t    = std::function<void(std::exception_ptr)>;
    using on_complete_fn_t = std::function<void()>;

    RxSinkComponent()
    {
        init_edge();
    }

    RxSinkComponent(std::string name = std::string()) : m_name(std::move(name))
    {
        init_edge();
    }

    ~RxSinkComponent() = default;

    template <typename... ArgsT>
    RxSinkComponent(ArgsT&&... args) : RxSinkComponent()
    {
        set_observer(std::forward<ArgsT>(args)...);
    }

    template <typename... ArgsT>
    RxSinkComponent(std::string name, ArgsT&&... args) : RxSinkComponent(std::move(name))
    {
        set_observer(std::forward<ArgsT>(args)...);
    }

    void set_observer(observer_t observer);

    template <typename... ArgsT>
    void set_observer(ArgsT&&... args)
    {
        set_observer(rxcpp::make_observer_dynamic<T>(std::forward<ArgsT>(args)...));
    }

  private:
    void init_edge()
    {
        auto edge = std::make_shared<EdgeRxObserver<T>>();

        m_sink_edge = edge;

        WritableProvider<T>::init_owned_edge(edge);
    }

    std::weak_ptr<EdgeRxObserver<T>> m_sink_edge;
    std::string m_name;
    // observer_t m_observer;
};

template <typename T>
void RxSinkComponent<T>::set_observer(rxcpp::observer<T> observer)
{
    if (auto edge = m_sink_edge.lock())
    {
        edge->set_observer(observer);
    }
    else
    {
        LOG(ERROR) << "Edge has expired";
    }
}

}  // namespace mrc::node
