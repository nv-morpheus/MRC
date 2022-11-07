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

#include "srf/experimental/modules/module_registry_util.hpp"
#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/node/operators/broadcast.hpp"
#include "srf/segment/builder.hpp"
#include "srf/version.hpp"

#include <nlohmann/json.hpp>

#include <atomic>

// TODO(Devin): Should be connected to a DataBufferModule
namespace srf::modules {
template <typename DataTypeT>
class MirrorTapModule : public SegmentModule
{
  public:
    MirrorTapModule(std::string module_name);

    MirrorTapModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    static std::atomic<unsigned int> s_tap_index;
    std::string m_egress_name{"mirror_tap"};
};

template <typename DataTypeT>
std::atomic<unsigned int> MirrorTapModule<DataTypeT>::s_tap_index{0};

template <typename DataTypeT>
MirrorTapModule<DataTypeT>::MirrorTapModule(std::string module_name) : SegmentModule(std::move(module_name))
{}

template <typename DataTypeT>
MirrorTapModule<DataTypeT>::MirrorTapModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

template <typename DataTypeT>
void MirrorTapModule<DataTypeT>::initialize(segment::Builder& builder)
{
    // ********** Process config ************ //
    if (config().contains("mirror_tap_egress"))
    {
        m_egress_name = config()["mirror_tap_egress"];
    }

    // ********** Implementation ************ //
    auto input =
        builder.template make_node<DataTypeT>("in", rxcpp::operators::map([](DataTypeT input) { return input; }));

    // Create deep-copy broadcast node.
    auto bcast = std::make_shared<node::Broadcast<DataTypeT>>(true);

    builder.make_edge(input, *bcast);

    auto output =
        builder.template make_node<DataTypeT>("in", rxcpp::operators::map([](DataTypeT input) { return input; }));

    builder.make_edge(bcast->make_source(), output);  // To next stage
    builder.make_edge(bcast->make_source(),
                      builder.get_egress<DataTypeT>(m_egress_name));// to mirror tap

    // Register the submodules output as one of this module's outputs
    register_input_port("in", input);
    register_output_port("out", output);
}

static MirrorTapModule<std::string> tap("test", {});
}  // namespace srf::modules