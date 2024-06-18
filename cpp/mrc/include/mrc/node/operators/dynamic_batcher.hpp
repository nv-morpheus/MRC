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
#include "rxcpp/operators/rx-observe_on.hpp"

#include <glog/logging.h>
#include <rxcpp/operators/rx-buffer_time_count.hpp>
#include <rxcpp/rx.hpp>

#include <exception>
#include <memory>
#include <mutex>

namespace mrc::node {
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
  DynamicBatcher(size_t max_count, std::chrono::milliseconds duration)
      : m_max_count(max_count), m_duration(duration) {
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
    // T input_data;
    // auto status = this->get_readable_edge()->await_read(input_data);

    // Create an observable from the input channel
    auto input_observable =
        rxcpp::observable<>::create<T>([this](rxcpp::subscriber<T> s) {
          T input_data;
          while (this->get_readable_edge()->await_read(input_data) ==
                 mrc::channel::Status::success) {
            s.on_next(input_data);
          }
          s.on_completed();
        });

    // DVLOG(1) << "DynamicBatcher: m_duration: " << m_duration.count() << std::endl;

    // Buffer the items from the input observable
    auto buffered_observable = input_observable.buffer_with_time_or_count(
        m_duration, m_max_count, rxcpp::observe_on_event_loop());

    // Subscribe to the buffered observable
    buffered_observable.subscribe(
        [this](const std::vector<T> &buffer) {
          this->get_writable_edge()->await_write(buffer);
        },
        []() {
          // Handle completion
        });

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
  void on_state_update(const state_t &state) final {
    switch (state) {
    case state_t::Stop:
      // Do nothing, we wait for the upstream channel to return closed
      // m_stop_source.request_stop();
      break;

    case state_t::Kill:
      m_stop_source.request_stop();
      break;

    default:
      break;
    }
  }

  std::stop_source m_stop_source;
  int m_max_count;
  std::chrono::milliseconds m_duration;
};
} // namespace mrc::node
