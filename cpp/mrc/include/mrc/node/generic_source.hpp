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
#include "mrc/node/rx_source.hpp"
#include "mrc/node/rx_subscribable.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace mrc::node {

template <typename T, typename ContextT>
class GenericSource : public RxSource<T, ContextT>
{
  public:
    GenericSource();
    ~GenericSource() override = default;

  private:
    using RxSource<T, ContextT>::set_observable;
    virtual void data_source(rxcpp::subscriber<T>& t) = 0;
};

template <typename T, typename ContextT>
GenericSource<T, ContextT>::GenericSource() :
  RxSource<T, ContextT>(rxcpp::observable<>::create<T>([this](rxcpp::subscriber<T> s) {
      try
      {
          data_source(s);

      } catch (...)
      {
          s.on_error(std::current_exception());
          return;
      }
      s.on_completed();
  }))
{
    // RxSource<T, ContextT>::set_observable();
}

}  // namespace mrc::node
