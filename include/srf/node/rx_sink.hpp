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
#include "srf/channel/status.hpp"
#include "srf/constants.hpp"
#include "srf/core/utils.hpp"
#include "srf/core/watcher.hpp"
#include "srf/exceptions/runtime_error.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/rx_prologue_tap.hpp"
#include "srf/node/rx_runnable.hpp"
#include "srf/node/rx_sink_base.hpp"
#include "srf/node/rx_subscribable.hpp"
#include "srf/node/sink_channel.hpp"
#include "srf/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-subscription.hpp>
#include <rxcpp/rx.hpp>

#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace srf::node {

// template <typename T, typename ContextT>
// class RxSink : public RxSinkBase<T>, public RxRunnable<ContextT>, public RxPrologueTap<T>
// {
//   public:
//     using observer_t       = rxcpp::observer<T>;
//     using on_next_fn_t     = std::function<void(T)>;
//     using on_error_fn_t    = std::function<void(std::exception_ptr)>;
//     using on_complete_fn_t = std::function<void()>;

//     RxSink()           = default;
//     ~RxSink() override = default;

//     template <typename... ArgsT>
//     RxSink(ArgsT&&... args)
//     {
//         set_observer(std::forward<ArgsT>(args)...);
//     }

//     void set_observer(observer_t observer);

//     template <typename... ArgsT>
//     void set_observer(ArgsT&&... args)
//     {
//         set_observer(rxcpp::make_observer_dynamic<T>(std::forward<ArgsT>(args)...));
//     }

//   private:
//     // the following methods are moved to private from their original scopes to prevent access from deriving classes
//     using RxSinkBase<T>::observable;

//     void on_shutdown_critical_section() final;
//     void do_subscribe(rxcpp::composite_subscription& subscription) final;

//     void on_stop(const rxcpp::subscription& subscription) const final;
//     void on_kill(const rxcpp::subscription& subscription) const final;

//     observer_t m_observer;
// };

// template <typename T, typename ContextT>
// void RxSink<T, ContextT>::set_observer(rxcpp::observer<T> observer)
// {
//     m_observer = std::move(observer);
// }

// template <typename T, typename ContextT>
// void RxSink<T, ContextT>::do_subscribe(rxcpp::composite_subscription& subscription)
// {
//     auto observable = RxPrologueTap<T>::apply_prologue_taps(RxSinkBase<T>::observable());

//     auto default_error_handler = rxcpp::make_observer_dynamic<T>(
//         [this](T data) { m_observer.on_next(std::move(data)); },
//         [this](std::exception_ptr ptr) {
//             runnable::Context::get_runtime_context().set_exception(std::move(std::current_exception()));
//             try
//             {
//                 m_observer.on_error(std::move(ptr));
//             } catch (...)
//             {
//                 runnable::Context::get_runtime_context().set_exception(std::move(std::current_exception()));
//             }
//         },
//         [this] { m_observer.on_completed(); });

//     observable.subscribe(subscription, default_error_handler);
// }

// template <typename T, typename ContextT>
// void RxSink<T, ContextT>::on_stop(const rxcpp::subscription& subscription) const
// {}

// template <typename T, typename ContextT>
// void RxSink<T, ContextT>::on_kill(const rxcpp::subscription& subscription) const
// {
//     subscription.unsubscribe();
// }

// template <typename T, typename ContextT>
// void RxSink<T, ContextT>::on_shutdown_critical_section()
// {}

template <typename T>
class EdgeRxObserver : public EdgeWritable<T>
{
  public:
    using observer_t = rxcpp::observer<T>;

    EdgeRxObserver() {}

    ~EdgeRxObserver()
    {
        m_observer.on_completed();
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

    RxSink()           = default;
    ~RxSink() override = default;

    template <typename... ArgsT>
    RxSink(ArgsT&&... args)
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

    void on_stop(const rxcpp::subscription& subscription) const final;
    void on_kill(const rxcpp::subscription& subscription) const final;

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
        [this](T data) { m_observer.on_next(std::move(data)); },
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
        [this] { m_observer.on_completed(); });

    observable.subscribe(subscription, default_error_handler);
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_stop(const rxcpp::subscription& subscription) const
{}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_kill(const rxcpp::subscription& subscription) const
{
    subscription.unsubscribe();
}

template <typename T, typename ContextT>
void RxSink<T, ContextT>::on_shutdown_critical_section()
{}

template <typename T>
class RxSinkComponent : public IngressProvider<T>
{
  public:
    using observer_t       = rxcpp::observer<T>;
    using on_next_fn_t     = std::function<void(T)>;
    using on_error_fn_t    = std::function<void(std::exception_ptr)>;
    using on_complete_fn_t = std::function<void()>;

    RxSinkComponent()
    {
        auto edge = std::make_shared<EdgeRxObserver<T>>();

        m_sink_edge = edge;

        IngressProvider<T>::init_edge(edge);
    }

    ~RxSinkComponent() = default;

    template <typename... ArgsT>
    RxSinkComponent(ArgsT&&... args)
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
    std::weak_ptr<EdgeRxObserver<T>> m_sink_edge;
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

}  // namespace srf::node
