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
#include "srf/node/edge.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/rx_epilogue_tap.hpp"
#include "srf/node/rx_runnable.hpp"
#include "srf/node/rx_source_base.hpp"
#include "srf/node/rx_subscribable.hpp"
#include "srf/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace srf::node {

/**
 * @brief Fully capable Source node that is runnable via RxRunnable
 *
 * @tparam T
 */
template <typename T, typename ContextT>
class RxSource : public RxSourceBase<T>, public RxRunnable<ContextT>, public RxEpilogueTap<T>
{
  public:
    RxSource() = default;
    RxSource(rxcpp::observable<T> observable);
    ~RxSource() override = default;

    void set_observable(rxcpp::observable<T> observable);

  private:
    void on_shutdown_critical_section() final;
    void do_subscribe(rxcpp::composite_subscription& subscription) final;
    void on_stop(const rxcpp::subscription& subscription) override;
    void on_kill(const rxcpp::subscription& subscription) final;

    rxcpp::observable<T> m_observable;
};

template <typename T, typename ContextT>
RxSource<T, ContextT>::RxSource(rxcpp::observable<T> observable)
{
    set_observable(observable);
}

template <typename T, typename ContextT>
void RxSource<T, ContextT>::on_shutdown_critical_section()
{
    DVLOG(10) << runnable::Context::get_runtime_context().info() << " releasing source channel";
    RxSourceBase<T>::release_channel();
}

template <typename T, typename ContextT>
void RxSource<T, ContextT>::set_observable(rxcpp::observable<T> observable)
{
    m_observable = std::move(observable);
}

template <typename T, typename ContextT>
void RxSource<T, ContextT>::do_subscribe(rxcpp::composite_subscription& subscription)
{
    auto observable = this->apply_epilogue_taps(m_observable);
    observable.subscribe(subscription, RxSourceBase<T>::observer());
}

template <typename T, typename ContextT>
void RxSource<T, ContextT>::on_stop(const rxcpp::subscription& subscription)
{
    subscription.unsubscribe();
}

template <typename T, typename ContextT>
void RxSource<T, ContextT>::on_kill(const rxcpp::subscription& subscription)
{
    subscription.unsubscribe();
}

}  // namespace srf::node
