/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "mrc/runnable/runnable.hpp"
#include "mrc/utils/type_utils.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>

template <typename T, typename ContextT>
class DynamicBatcher : public mrc::node::WritableProvider<T>,
                       public mrc::node::ReadableAcceptor<T>,
                       public mrc::node::SinkChannelOwner<T>,
                       public mrc::node::WritableAcceptor<std::vector<T>>,
                       public mrc::node::ReadableProvider<std::vector<T>>,
                       public mrc::node::SourceChannelOwner<std::vector<T>>,
                       public mrc::runnable::RunnableWithContext<ContextT> {
  using state_t = mrc::runnable::Runnable::State;
  using input_t = T;
  using output_t = std::vector<T>;

public:
  DynamicBatcher(size_t max_count) {
    // Set the default channel
    mrc::node::SinkChannelOwner<input_t>::set_channel(
        std::make_unique<mrc::channel::BufferedChannel<input_t>>());
    mrc::node::SourceChannelOwner<output_t>::set_channel(
        std::make_unique<mrc::channel::BufferedChannel<output_t>>());
  }
  ~DynamicBatcher() override = default;

private:
  /**
   * @brief Runnable's entrypoint.
   */
  void run(mrc::runnable::Context &ctx) override {
    T input_data;
    auto status = this->get_readable_edge()->await_read(input_data);

    // TODO(Yuchen): fill out the implementation here





    // Only drop the output edges if we are rank 0
    if (ctx.rank() == 0) {
      // Need to drop the output edges
      mrc::node::SourceProperties<output_t>::release_edge_connection();
      mrc::node::SinkProperties<T>::release_edge_connection();
    }
  }

  /**
   * @brief Runnable's state control, for stopping from MRC.
   */
  void on_state_update(const state_t &state) final;

  std::stop_source m_stop_source;
};
