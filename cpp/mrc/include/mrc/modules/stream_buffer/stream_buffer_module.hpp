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
    /**
     * @brief A simple stream buffer that can be used to capture the output of a stream.
     * @note This class is not thread safe.
     * @tparam T
     * @tparam N
     */
    template<typename T, std::size_t N>
    class SimpleSlidingWindow {
    public:
        SimpleSlidingWindow() : m_head(0), m_tail(0) {}

        void push(T element) {
            if (full()) {
                m_head = (m_abs_head_offset++) % N;
                m_dropped++;
            }
            m_buffer[m_tail] = element;
            m_tail = (m_tail + 1) % N;
        }

        T pop() {
            if (empty()) {
                throw std::runtime_error("RingBuffer is empty");
            }
            T element = m_buffer[m_head];
            m_head = (++m_abs_head_offset) % N;

            return element;
        }

        T peek() {
            T element = m_buffer[m_head];

            return element;
        }

        size_t size() const {
            if (m_tail >= m_head) {
                return m_tail - m_head;
            }

            return N - m_head + m_tail;
        }

        bool empty() const {
            return m_head == m_tail;
        }

        bool full() const {
            return (m_tail + 1) % N == m_head;
        }

    private:
        std::array<T, N> m_buffer;

        std::uint64_t m_dropped{0};
        std::uint64_t m_abs_head_offset{0};

        std::size_t m_head;
        std::size_t m_tail;
    };

    template<typename DataTypeT>
    class SimpleImmediateStreamBuffer : public SegmentModule {
        using type_t = SimpleImmediateStreamBuffer<DataTypeT>;

    public:
        SimpleImmediateStreamBuffer(std::string module_name);

        SimpleImmediateStreamBuffer(std::string module_name, nlohmann::json config);

    protected:
        void initialize(segment::Builder &builder) override;

        std::string module_type_name() const override;

    private:
        static std::atomic<unsigned int> s_instance_index;

        boost::circular_buffer<DataTypeT> m_ring_buffer;
        rxcpp::subjects::subject<DataTypeT> m_subject{};
        rxcpp::observable<DataTypeT> m_observable;
        rxcpp::observer<DataTypeT> m_observer;

        std::string m_ingress_name;

        rxcpp::observable<DataTypeT> create_observable();
    };

    template<typename DataTypeT>
    SimpleImmediateStreamBuffer<DataTypeT>::SimpleImmediateStreamBuffer(std::string module_name) : SegmentModule(
            std::move(module_name)), m_ring_buffer(128) {
        auto hot_obs = m_subject.get_observable().publish();
        hot_obs.connect();
        m_observable = hot_obs;
    }


    template<typename DataTypeT>
    SimpleImmediateStreamBuffer<DataTypeT>::SimpleImmediateStreamBuffer(std::string module_name, nlohmann::json config)
            : SegmentModule(
            std::move(module_name), std::move(config)), m_ring_buffer{128} {
        auto hot_obs = m_subject.get_observable().publish();
        hot_obs.connect();
        m_observable = hot_obs;
    }

    template<typename DataTypeT>
    rxcpp::observable<DataTypeT> SimpleImmediateStreamBuffer<DataTypeT>::create_observable() {
        return rxcpp::observable<>::create<DataTypeT>([this](rxcpp::subscriber<DataTypeT> &subscriber) {
            while (subscriber.is_subscribed()) {
                if (!m_ring_buffer.empty()) {
                    subscriber.on_next(m_ring_buffer.front());
                    m_ring_buffer.pop_front();
                } else {
                    boost::this_fiber::yield();
                }
            }
        });
    }

    template<typename DataTypeT>
    void SimpleImmediateStreamBuffer<DataTypeT>::initialize(segment::Builder &builder) {
        auto buffer_sink = builder.template make_sink<DataTypeT>("buffer_sink_new", m_subject.get_subscriber());

        //auto buffer_sink_old = builder.template make_sink<DataTypeT>("buffer_sink_old", [this](DataTypeT data) {
        //    // Non-blocking, no fail: we just overwrite data if we run out of space, don't block the ingress queue
        //    // under any circumstances.
        //    std::cerr << "Got input data: " << data << std::endl;
        //    //m_ring_buffer.push_back(data);
        //    if (m_subject.has_observers()) {
        //        m_subject.get_subscriber().on_next(std::move(data));
        //    }
        //});

        m_observable.subscribe([](DataTypeT data) {
            std::cerr << "***Got data 1: " << data << std::endl;
        });
        m_observable.subscribe([](DataTypeT data) {
            std::cerr << "***Got data 2: " << data << std::endl;
        });
        m_observable.subscribe([](DataTypeT data) {
            std::cerr << "***Got data 3: " << data << std::endl;
        });

        //auto buffer_source_new = builder.template make_source<DataTypeT>("buffer_source_new", m_observable);

        // This should be resolved by Michael's non-linear pipeline updates.
        // Currently, no way to know when to shut down.
        auto buffer_source_old = builder.template make_source<DataTypeT>(
                "buffer_source_old",
                [this](rxcpp::subscriber<DataTypeT> &subscriber) {
                    if (subscriber.is_subscribed()) {
                        subscriber.on_next(std::to_string(this->m_subject.has_observers()));
                        subscriber.on_next(std::to_string(this->m_subject.has_observers()));
                        subscriber.on_next(std::to_string(this->m_subject.has_observers()));
                        subscriber.on_next(std::to_string(this->m_subject.has_observers()));
                        boost::this_fiber::sleep_for(std::chrono::seconds(1));
                    }

                    //m_observable = create_observable();
                    //auto hot_obs = m_observable.publish();
                    //hot_obs.subscribe([&subscriber](DataTypeT data) {
                    //    if (subscriber.is_subscribed()) {
                    //        subscriber.on_next(data);
                    //    }
                    //});
                    //hot_obs.connect();

                    subscriber.on_completed();
                });

        register_input_port("input", buffer_sink);
        register_output_port("output", buffer_source_old);
    }

    template<typename DataTypeT>
    std::string SimpleImmediateStreamBuffer<DataTypeT>::module_type_name() const {
        return std::string(::mrc::type_name<type_t>());
    }
}