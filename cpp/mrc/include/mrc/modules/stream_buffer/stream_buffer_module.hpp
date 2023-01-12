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

#include "mrc/modules/mirror_tap/mirror_tap_module.hpp"
#include "mrc/modules/segment_modules.hpp"

#include <boost/circular_buffer.hpp>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>

#include <cstddef>
#include <string>
#include <deque>
#include <iostream>

namespace mrc::modules {

    template<typename DataTypeT>
    class SimpleImmediateStreamBuffer : public SegmentModule {
        using type_t = SimpleImmediateStreamBuffer<DataTypeT>;

    public:
        SimpleImmediateStreamBuffer(std::string module_name);

        SimpleImmediateStreamBuffer(std::string module_name, nlohmann::json config);

        ~SimpleImmediateStreamBuffer() {
            std::cerr << "Destructing SimpleImmediateStreamBuffer" << std::endl;
        }

    protected:
        void initialize(segment::Builder &builder) override;

        std::string module_type_name() const override;

    private:
        static std::atomic<unsigned int> s_instance_index;

        boost::circular_buffer<DataTypeT> m_ring_buffer;
        rxcpp::subjects::subject<DataTypeT> m_subject{};

        std::string m_ingress_name;
    };

    template<typename DataTypeT>
    SimpleImmediateStreamBuffer<DataTypeT>::SimpleImmediateStreamBuffer(std::string module_name)
            :   SegmentModule(std::move(module_name)),
                m_ring_buffer(128) {
    }


    template<typename DataTypeT>
    SimpleImmediateStreamBuffer<DataTypeT>::SimpleImmediateStreamBuffer(std::string module_name, nlohmann::json config)
            :   SegmentModule(std::move(module_name), std::move(config)),
                m_ring_buffer{128} {
    }

    template<typename DataTypeT>
    void SimpleImmediateStreamBuffer<DataTypeT>::initialize(segment::Builder &builder) {
        auto buffer_sink = builder.template make_sink<DataTypeT>("buffer_sink_new", m_subject.get_subscriber());

        // This is a hack, because we don't correctly support passing observables to RxSource creation yet
        // Consume values from subject and push them to ring buffer
        m_subject.get_observable().subscribe(
                [this](DataTypeT data) {
                    m_ring_buffer.push_back(std::move(data));
                    VLOG(10) << "Subscriber 1: OnNext -> push to ring buffer: " << data << std::endl;
                },
                [this](std::exception_ptr ep) {
                    VLOG(10) << "Subscriber 1: OnError" << std::endl;
                },
                [this]() {
                    VLOG(10) << "Subscriber 1: OnCompleted" << std::endl;
                }
        );

        // Example of adding a second subscriber
        m_subject.get_observable().subscribe(
                [this](DataTypeT data) {
                    VLOG(10) << "Subscriber 2: OnNext -> " << data << std::endl;
                },
                [this](std::exception_ptr ep) {
                    VLOG(10) << "Subscriber 2: OnError" << std::endl;
                },
                [this]() {
                    VLOG(10) << "Subscriber 2: OnCompleted" << std::endl;
                }
        );

        // Create our source that reads from the buffer as long as it has a subscriber and
        // our subject hasn't called on_complete()
        auto buffer_source = builder.template make_source<DataTypeT>(
                "buffer_source",
                [this](rxcpp::subscriber<DataTypeT> &subscriber) {
                    // m_subject.get_observable().subscribe(subscriber);
                    while (subscriber.is_subscribed() && m_subject.has_observers()) {
                        if (!m_ring_buffer.empty()) {
                            subscriber.on_next(m_ring_buffer.front());
                            m_ring_buffer.pop_front();
                        } else {
                            boost::this_fiber::yield();
                        }
                    }
                });

        register_input_port("input", buffer_sink);
        register_output_port("output", buffer_source);
    }

    template<typename DataTypeT>
    std::string SimpleImmediateStreamBuffer<DataTypeT>::module_type_name() const {
        return std::string(::mrc::type_name<type_t>());
    }
}