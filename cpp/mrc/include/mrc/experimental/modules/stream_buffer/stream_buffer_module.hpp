/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/experimental/modules/mirror_tap/mirror_tap.hpp"
#include "mrc/experimental/modules/stream_buffer/stream_buffer_immediate.hpp"
#include "mrc/experimental/modules/stream_buffer/stream_buffer_traits.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/segment/builder.hpp"

#include <boost/circular_buffer.hpp>
#include <boost/fiber/all.hpp>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <deque>
#include <iostream>
#include <string>
#include <type_traits>

namespace mrc::modules {

/**
 * @brief Buffers a stream of data; avoiding any stalls by over-writing older data when the buffer is full.
 * We guarantee not to block the data stream; but, may drop data if the buffer is full.
 * @tparam DataTypeT The type of data to buffer
 */
template <typename DataTypeT,
          template <typename> class StreamBufferTypeT = mrc::modules::stream_buffers::StreamBufferImmediate>
class StreamBufferModule : public SegmentModule, public PersistentModule
{
    static_assert(stream_buffers::IsStreamBuffer<DataTypeT, StreamBufferTypeT>,
                  "StreamBufferTypeT must be derived from StreamBufferBase");
    using type_t = StreamBufferModule<DataTypeT, StreamBufferTypeT>;

  public:
    StreamBufferModule(std::string module_name);

    StreamBufferModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::IBuilder& builder) override;

    std::string module_type_name() const override;

  private:
    StreamBufferTypeT<DataTypeT> m_stream_buffer;
    rxcpp::subjects::subject<DataTypeT> m_subject{};
};

template <typename DataTypeT, template <typename> class StreamBufferTypeT>
StreamBufferModule<DataTypeT, StreamBufferTypeT>::StreamBufferModule(std::string module_name) :
  SegmentModule(std::move(module_name)),
  PersistentModule(),
  m_stream_buffer()
{}

template <typename DataTypeT, template <typename> class StreamBufferTypeT>
StreamBufferModule<DataTypeT, StreamBufferTypeT>::StreamBufferModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config)),
  PersistentModule(),
  m_stream_buffer()
{
    if (this->config().contains("buffer_size"))
    {
        m_stream_buffer.buffer_size(this->config()["buffer_size"]);
    }
}

template <typename DataTypeT, template <typename> class StreamBufferTypeT>
void StreamBufferModule<DataTypeT, StreamBufferTypeT>::initialize(segment::IBuilder& builder)
{
    auto buffer_sink = builder.template make_sink<DataTypeT>("buffer_sink_new", m_subject.get_subscriber());

    // This is a hack, because we don't correctly support passing observables to RxSource creation yet
    // Consume values from subject and push them to ring buffer
    m_subject.get_observable().subscribe(
        [this](DataTypeT data) {
            m_stream_buffer.push_back(std::move(data));
            VLOG(10) << "Subscriber 1: OnNext -> push to ring buffer " << std::endl;
        },
        [this](std::exception_ptr ep) {
            VLOG(10) << "Subscriber 1: OnError" << std::endl;
        },
        [this]() {
            VLOG(10) << "Subscriber 1: OnCompleted" << std::endl;
        });

    // Example of adding a second subscriber
    /*
    m_subject.get_observable().subscribe(
        [this](DataTypeT data) {
            VLOG(10) << "Subscriber 2: OnNext -> " << data << std::endl;
        },
        [this](std::exception_ptr ep) {
            VLOG(10) << "Subscriber 2: OnError" << std::endl;
        },
        [this]() {
            VLOG(10) << "Subscriber 2: OnCompleted" << std::endl;
        });
    */

    // Create our source that reads from the buffer as long as it has a subscriber and
    // our subject hasn't called on_complete()
    auto buffer_source = builder.template make_source<DataTypeT>(
        "buffer_source",
        [this](rxcpp::subscriber<DataTypeT>& subscriber) {
            // TODO(Devin): not currently supported
            // m_subject.get_observable().subscribe(subscriber);
            while (subscriber.is_subscribed() && m_subject.has_observers())
            {
                if (!m_stream_buffer.empty())
                {
                    m_stream_buffer.flush_all(subscriber);
                }
                else
                {
                    boost::this_fiber::yield();
                }
            }
        });

    builder.register_module_input("input", buffer_sink);
    builder.register_module_output("output", buffer_source);
}

template <typename DataTypeT, template <typename> class StreamBufferTypeT>
std::string StreamBufferModule<DataTypeT, StreamBufferTypeT>::module_type_name() const
{
    return std::string(::mrc::type_name<type_t>());
}
}  // namespace mrc::modules
