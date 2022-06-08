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

#include <srf/channel/buffered_channel.hpp>
#include <srf/channel/channel.hpp>
#include <srf/channel/status.hpp>
#include <srf/constants.hpp>
#include <srf/core/utils.hpp>
#include <srf/core/watcher.hpp>
#include <srf/exceptions/runtime_error.hpp>
#include <srf/node/edge.hpp>
#include <srf/node/forward.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/rx_sink_base.hpp>
#include <srf/node/sink_channel.hpp>
#include <srf/utils/type_utils.hpp>

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace srf::node {

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

}  // namespace srf::node
