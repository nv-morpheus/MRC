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
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/sink_channel.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace mrc::node {

template <typename InputT, typename OutputT, typename ContextT>
class GenericNode : public RxNode<InputT, OutputT, ContextT>
{
  public:
    GenericNode() :
      RxNode<InputT, OutputT, ContextT>([this](const rxcpp::observable<InputT>& input) {
          return rxcpp::observable<>::create<OutputT>([this, input](rxcpp::subscriber<OutputT> output) {
              input.subscribe([this, &output](InputT i) { on_data(std::move(i), output); },
                              [this, &output] { on_completed(output); });
          });
      })
    {}

  private:
    virtual void on_data(InputT&& data, rxcpp::subscriber<OutputT>& subscriber) = 0;
    virtual void on_completed(rxcpp::subscriber<OutputT>& subscriber) {}
};

}  // namespace mrc::node
