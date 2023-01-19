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

#include "mrc/modules/mirror_tap/mirror_tap_sink.hpp"
#include "mrc/modules/module_registry_util.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/version.hpp"

#include <nlohmann/json.hpp>

#include <atomic>

namespace mrc::modules {
    template<typename DataTypeT>
    class MirrorTapSourceModule : public SegmentModule, public PersistentModule,
                                  public std::enable_shared_from_this<MirrorTapSourceModule<DataTypeT>> {
        using type_t = MirrorTapSourceModule<DataTypeT>;

    public:
        MirrorTapSourceModule(std::string module_name);

        MirrorTapSourceModule(std::string module_name, nlohmann::json config);

        segment::IngressPorts<DataTypeT> create_ingress_ports() const;

        segment::EgressPorts<DataTypeT> create_egress_ports() const;

        std::string get_port_name() const;

        std::shared_ptr<SegmentModule> get_sink() const;

        // TODO: Make higher level MirrorTap that creates source and sink elements.
        template<typename FunctionT>
        auto tap_segment(FunctionT initializer, const std::string tap_from, const std::string tap_to) {
            using namespace modules;
            return [this, initializer, tap_from, tap_to](segment::Builder &builder) {
                initializer(builder);
                builder.init_module(get_shared_ptr());

                builder.make_edge_tap<DataTypeT>(tap_from, tap_to,
                                                 input_port("input"),
                                                 output_port("output"));
            };
        }

        template<typename FunctionT>
        auto attach_tap_output(FunctionT initializer, const std::string entry_point) {
            using namespace modules;
            return [this, initializer, entry_point](segment::Builder &builder) {
                initializer(builder);

                builder.init_module(m_sink);
                builder.make_edge<DataTypeT>(m_sink->output_port("output"), entry_point);
            };
        }


    protected:
        void initialize(segment::Builder &builder) override;

        std::string module_type_name() const override;

    private:
        static std::atomic<unsigned int> s_tap_index;

        std::shared_ptr<MirrorTapSourceModule<DataTypeT>> get_shared_ptr() {
            return this->shared_from_this();
        }

        std::shared_ptr<MirrorTapSinkModule<DataTypeT>> m_sink;

        std::string m_egress_name;
    };

    template<typename DataTypeT>
    std::atomic<unsigned int> MirrorTapSourceModule<DataTypeT>::s_tap_index{0};

    template<typename DataTypeT>
    MirrorTapSourceModule<DataTypeT>::MirrorTapSourceModule(std::string module_name)
            : SegmentModule(std::move(module_name)) {
        m_egress_name = "mirror_tap_" + std::to_string(s_tap_index++);

        m_sink = std::make_shared<MirrorTapSinkModule<DataTypeT>>(name() + "_sink");
        m_sink->m_ingress_name = m_egress_name;
    }

    // TODO
    template<typename DataTypeT>
    MirrorTapSourceModule<DataTypeT>::MirrorTapSourceModule(std::string module_name, nlohmann::json _config) :
            SegmentModule(std::move(module_name), std::move(_config)) {
        m_egress_name = "mirror_tap_" + std::to_string(s_tap_index++);

        m_sink = std::make_shared<MirrorTapSinkModule<DataTypeT>>(name() + "_sink", config());
        m_sink->m_ingress_name = m_egress_name;
    }

    template<typename DataTypeT>
    std::string MirrorTapSourceModule<DataTypeT>::get_port_name() const {
        return m_egress_name;
    }

    template<typename DataTypeT>
    std::shared_ptr<SegmentModule> MirrorTapSourceModule<DataTypeT>::get_sink() const {
        return m_sink;
    }

    template<typename DataTypeT>
    [[maybe_unused]] segment::EgressPorts<DataTypeT> MirrorTapSourceModule<DataTypeT>::create_egress_ports() const {
        return segment::EgressPorts<DataTypeT>({m_egress_name});
    }

    template<typename DataTypeT>
    [[maybe_unused]] segment::IngressPorts<DataTypeT> MirrorTapSourceModule<DataTypeT>::create_ingress_ports() const {
        return segment::IngressPorts<DataTypeT>({m_egress_name});
    }

    template<typename DataTypeT>
    void MirrorTapSourceModule<DataTypeT>::initialize(segment::Builder &builder) {
        // ********** Implementation ************ //
        auto input = builder.construct_object<node::Broadcast<DataTypeT>>("broadcast");

        auto output = builder.template make_node<DataTypeT>("output",
                                                            rxcpp::operators::tap([](DataTypeT input) {}));

        builder.make_edge(input, builder.get_egress<DataTypeT>(m_egress_name));  // to mirror tap
        builder.make_edge(input, output);  // To next stage

        // Register the submodules output as one of this module's outputs
        register_input_port("input", input);
        register_output_port("output", output);
    }

    template<typename DataTypeT>
    std::string MirrorTapSourceModule<DataTypeT>::module_type_name() const {
        return std::string(::mrc::type_name<type_t>());
    }
}  // namespace mrc::modules