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
#include "srf/segment/builder.hpp"
#include "srf/version.hpp"

#include <nlohmann/json.hpp>

namespace srf::modules {
template <typename DataTypeT>
class MirrorTapModule : public SegmentModule
{
  public:
    MirrorTapModule(std::string module_name);

    MirrorTapModule(std::string module_name, nlohmann::json config);

  protected:
    void initialize(segment::Builder& builder) override;
};

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

    // ********** Implementation ************ //
    auto tap_and_forward = builder.template make_node<DataTypeT>("in", rxcpp::operators::map([](DataTypeT input) {
                                                                     DataTypeT copy = input;

                                                                     return input;
                                                                 }));

    // Register the submodules output as one of this module's outputs
    register_input_port("in", tap_and_forward);
    register_output_port("out", tap_and_forward);
}

static MirrorTapModule<std::string> tap("test", {});
}  // namespace srf::modules