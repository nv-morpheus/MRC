/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/version.hpp"

#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include <atomic>

namespace mrc::modules {
template <typename DataTypeT>
class MirrorTapModule : public SegmentModule, public PersistentModule
{
    using type_t = MirrorTapModule<DataTypeT>;

  public:
    MirrorTapModule(std::string module_name);

    MirrorTapModule(std::string module_name, nlohmann::json config);

    std::string tap_egress_port_name() const;

  protected:
    void initialize(segment::Builder& builder) override;

    std::string module_type_name() const override;

  private:
    [[maybe_unused]] static std::atomic<unsigned int> s_tap_id;

    std::string m_egress_name;
};

template <typename DataTypeT>
std::atomic<unsigned int> MirrorTapModule<DataTypeT>::s_tap_id{0};

template <typename DataTypeT>
MirrorTapModule<DataTypeT>::MirrorTapModule(std::string module_name) :
  SegmentModule(std::move(module_name)),
  m_egress_name("mirror_tap_source_" + std::to_string(s_tap_id++))
{}

template <typename DataTypeT>
MirrorTapModule<DataTypeT>::MirrorTapModule(std::string module_name, nlohmann::json _config) :
  SegmentModule(std::move(module_name), std::move(_config)),
  m_egress_name("mirror_tap_source_" + std::to_string(s_tap_id++))
{
    if (config().contains("tap_id_override"))
    {
        m_egress_name = config()["tap_id_override"];
    }
}

template <typename DataTypeT>
std::string MirrorTapModule<DataTypeT>::tap_egress_port_name() const
{
    return m_egress_name;
}

template <typename DataTypeT>
void MirrorTapModule<DataTypeT>::initialize(segment::Builder& builder)
{
    // ********** Implementation ************ //
    auto bcast = builder.construct_object<node::Broadcast<DataTypeT>>("broadcast");

    builder.make_edge(bcast, builder.get_egress<DataTypeT>(m_egress_name));  // to mirror tap

    // Register the submodules output as one of this module's outputs
    register_input_port("input", bcast);
    register_output_port("output", bcast);
}

template <typename DataTypeT>
std::string MirrorTapModule<DataTypeT>::module_type_name() const
{
    return std::string(::mrc::type_name<type_t>());
}
}  // namespace mrc::modules
