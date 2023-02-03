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

#include "mrc/modules/mirror_tap/mirror_tap.hpp"
#include "mrc/modules/mirror_tap/mirror_tap_stream.hpp"
#include "mrc/modules/module_registry_util.hpp"
#include "mrc/modules/properties/persistent.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/ingress_ports.hpp"
#include "mrc/version.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <atomic>

namespace mrc::modules {
template <typename DataTypeT>
class MirrorTapUtil
{
    using initializer_t = std::function<void(segment::Builder& builder)>;
    using type_t = MirrorTapUtil<DataTypeT>;

  public:
    MirrorTapUtil(std::string module_name);

    MirrorTapUtil(std::string module_name, nlohmann::json config);

    initializer_t tap(initializer_t initializer, const std::string tap_from, const std::string tap_to)
    {
        using namespace modules;
        return [this, initializer, tap_from, tap_to](segment::Builder& builder) {
            initializer(builder);
            builder.init_module(m_tap);

            builder.make_edge_tap<DataTypeT>(tap_from, tap_to, m_tap->input_port("input"), m_tap->output_port("output"));
        };
    }

    initializer_t stream_to(initializer_t initializer, const std::string entry_point)
    {
        using namespace modules;
        return [this, initializer, entry_point](segment::Builder& builder) {
            initializer(builder);

            builder.init_module(m_stream);
            builder.make_edge<DataTypeT>(m_stream->output_port("output"), entry_point);
        };
    }

    template <typename... IngressTypesT>
    segment::IngressPorts<IngressTypesT..., DataTypeT> create_or_extend_ingress_ports(
        segment::IngressPorts<IngressTypesT...>& ingress_ports) const
    {
        auto names(ingress_ports.names());
        names.push_back(get_ingress_tap_name());

        return segment::IngressPorts<IngressTypesT..., DataTypeT>(std::move(names));
    }

    segment::IngressPorts<DataTypeT> create_or_extend_ingress_ports() const
    {
        return segment::IngressPorts<DataTypeT>({get_ingress_tap_name()});
    }

    template <typename... EgressTypesT>
    segment::EgressPorts<EgressTypesT..., DataTypeT> create_or_extend_egress_ports(
        segment::EgressPorts<EgressTypesT...>& egress_ports) const
    {
        auto names(egress_ports.names());
        names.push_back(get_ingress_tap_name());

        return segment::EgressPorts<EgressTypesT..., DataTypeT>(std::move(names));
    }

    segment::EgressPorts<DataTypeT> create_or_extend_egress_ports() const
    {
        return segment::EgressPorts<DataTypeT>({get_egress_tap_name()});
    }

    std::string get_egress_tap_name() const;

    std::string get_ingress_tap_name() const;

  private:
    std::shared_ptr<MirrorTapModule<DataTypeT>> m_tap;
    std::shared_ptr<MirrorTapStreamModule<DataTypeT>> m_stream;
};

template <typename DataTypeT>
MirrorTapUtil<DataTypeT>::MirrorTapUtil(std::string tap_name) :
  m_tap(std::make_shared<MirrorTapModule<DataTypeT>>(std::move(tap_name))),
  m_stream(std::make_shared<MirrorTapStreamModule<DataTypeT>>(std::move(tap_name)))
{
    m_stream->tap_ingress_port_name(m_tap->tap_egress_port_name());
}

template <typename DataTypeT>
MirrorTapUtil<DataTypeT>::MirrorTapUtil(std::string tap_name, nlohmann::json config) :
  m_tap(std::make_shared<MirrorTapModule<DataTypeT>>(tap_name, config)),
  m_stream(std::make_shared<MirrorTapStreamModule<DataTypeT>>(tap_name, config))
{
    m_stream->tap_ingress_port_name(m_tap->tap_egress_port_name());
}

template <typename DataTypeT>
[[maybe_unused]] std::string MirrorTapUtil<DataTypeT>::get_egress_tap_name() const
{
    return m_tap->tap_egress_port_name();
}

template <typename DataTypeT>
[[maybe_unused]] std::string MirrorTapUtil<DataTypeT>::get_ingress_tap_name() const
{
    return m_stream->tap_ingress_port_name();
}
}  // namespace mrc::modules