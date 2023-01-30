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

#include "mrc/modules/module_registry_util.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/modules/stream_buffer/immediate_stream_buffer.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/version.hpp"

#include <nlohmann/json.hpp>

#include <atomic>

namespace mrc::modules {
    template<typename DataTypeT>
    class MirrorTapStreamModule : public SegmentModule, public PersistentModule {
        using type_t = MirrorTapStreamModule<DataTypeT>;

    public:
        MirrorTapStreamModule(std::string module_name);

        MirrorTapStreamModule(std::string module_name, nlohmann::json _config);

        std::string tap_ingress_port_name() const;
        void tap_ingress_port_name(std::string name);

    protected:
        void initialize(segment::Builder &builder) override;

        std::string module_type_name() const override;

    private:
        std::shared_ptr<ImmediateStreamBufferModule<DataTypeT>> m_stream_buffer;

        std::string m_ingress_name;
    };

    template<typename DataTypeT>
    MirrorTapStreamModule<DataTypeT>::MirrorTapStreamModule(std::string module_name)
            : SegmentModule(std::move(module_name)) {
    }

    template<typename DataTypeT>
    MirrorTapStreamModule<DataTypeT>::MirrorTapStreamModule(std::string module_name, nlohmann::json _config)
            : SegmentModule(std::move(module_name), std::move(_config)) {

        if (config().contains("tap_id_override"))
        {
            m_ingress_name = config()["tap_id_override"];
        }
    }

    template<typename DataTypeT>
    std::string MirrorTapStreamModule<DataTypeT>::tap_ingress_port_name() const {
        return m_ingress_name;
    }

    template<typename DataTypeT>
    void MirrorTapStreamModule<DataTypeT>::tap_ingress_port_name(std::string ingress_port_name) {
        m_ingress_name = std::move(ingress_port_name);
    }

    template<typename DataTypeT>
    void MirrorTapStreamModule<DataTypeT>::initialize(segment::Builder &builder) {
        auto mirror_ingress = builder.get_ingress<DataTypeT>(m_ingress_name);
        // TODO
        m_stream_buffer = builder.make_module<ImmediateStreamBufferModule<DataTypeT>>("test", {});

        builder.make_edge(mirror_ingress, m_stream_buffer->input_port("input"));

        register_output_port("output", m_stream_buffer->output_port("output"));
    }

    template<typename DataTypeT>
    std::string MirrorTapStreamModule<DataTypeT>::module_type_name() const {
        return std::string(::mrc::boost_type_name<type_t>());
    }
}