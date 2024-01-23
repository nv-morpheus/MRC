/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../test_mrc.hpp"  // IWYU pragma: keep

#include "mrc/modules/properties/persistent.hpp"
#include "mrc/modules/segment_modules.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"   // IWYU pragma: keep
#include "mrc/utils/type_utils.hpp"  // for type_name

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <rxcpp/rx.hpp>  // for subscriber, map

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>  // for move
// IWYU pragma: no_include "gtest/gtest_pred_impl.h"

namespace mrc {

class TestSegmentResources
{
  public:
    TestSegmentResources() = default;

    static std::unique_ptr<Options> make_options()
    {
        auto options = std::make_unique<Options>();
        options->topology().user_cpuset("0");
        return options;
    }
};

class TestModules : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline  = mrc::make_pipeline();
        m_resources = std::make_shared<TestSegmentResources>();
    }

    void TearDown() override {}

    std::unique_ptr<pipeline::IPipeline> m_pipeline;
    std::shared_ptr<TestSegmentResources> m_resources;
};

template <typename DataTypeT, std::size_t SourceCountV, std::size_t EmissionCountV>
class [[maybe_unused]] MultiSourceModule : public modules::SegmentModule
{
    using type_t = MultiSourceModule<DataTypeT, SourceCountV, EmissionCountV>;

  public:
    MultiSourceModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

    MultiSourceModule(std::string module_name, nlohmann::json config) :
      SegmentModule(std::move(module_name), std::move(config))
    {}

  protected:
    void initialize(segment::IBuilder& builder) override;

    std::string module_type_name() const override;
};

template <typename DataTypeT, std::size_t SourceCountV, std::size_t EmissionCountV>
void MultiSourceModule<DataTypeT, SourceCountV, EmissionCountV>::initialize(segment::IBuilder& builder)
{
    for (std::size_t i = 0; i < SourceCountV; ++i)
    {
        auto source = builder.make_source<DataTypeT>("source_" + std::to_string(i),
                                                     [](rxcpp::subscriber<DataTypeT>& subscriber) {
                                                         for (std::size_t i = 0; i < EmissionCountV; ++i)
                                                         {
                                                             subscriber.on_next(DataTypeT());
                                                         }

                                                         subscriber.on_completed();
                                                     });

        auto internal = builder.make_node<DataTypeT>("internal_" + std::to_string(i),
                                                     rxcpp::operators::map([](DataTypeT data) {
                                                         return data;
                                                     }));

        builder.make_edge(source, internal);

        auto output = builder.make_node<DataTypeT>("output_" + std::to_string(i),
                                                   rxcpp::operators::map([](DataTypeT data) {
                                                       return data;
                                                   }));

        builder.make_edge(internal, output);

        register_output_port("output_" + std::to_string(i), output);
    }
}

template <typename DataTypeT, std::size_t SinkCountV, std::size_t EmissionCountV>
std::string MultiSourceModule<DataTypeT, SinkCountV, EmissionCountV>::module_type_name() const
{
    return std::string(::mrc::type_name<type_t>());
}

template <typename DataTypeT, std::size_t SinkCountV>
class [[maybe_unused]] MultiSinkModule : public modules::SegmentModule, public modules::PersistentModule
{
    using type_t = MultiSinkModule<DataTypeT, SinkCountV>;

  public:
    MultiSinkModule(std::string module_name) : SegmentModule(std::move(module_name)) {}

    MultiSinkModule(std::string module_name, nlohmann::json config) :
      SegmentModule(std::move(module_name), std::move(config))
    {}

    std::size_t get_received(std::size_t index) const;

  protected:
    void initialize(segment::IBuilder& builder) override;

    std::string module_type_name() const override;

  private:
    std::array<std::size_t, SinkCountV> m_received_count{0};
};

template <typename DataTypeT, std::size_t SinkCountV>
std::size_t MultiSinkModule<DataTypeT, SinkCountV>::get_received(std::size_t index) const
{
    assert(index < m_received_count.size());
    return m_received_count[index];
}

template <typename DataTypeT, std::size_t SinkCountV>
void MultiSinkModule<DataTypeT, SinkCountV>::initialize(segment::IBuilder& builder)
{
    for (std::size_t i = 0; i < SinkCountV; ++i)
    {
        auto input = builder.make_sink<DataTypeT>("input_" + std::to_string(i), [this, i](DataTypeT data) {
            m_received_count[i]++;
        });

        register_input_port("input_" + std::to_string(i), input);
    }
}

template <typename DataTypeT, std::size_t SinkCountV>
std::string MultiSinkModule<DataTypeT, SinkCountV>::module_type_name() const
{
    return std::string(::mrc::type_name<type_t>());
}

using TestMirrorTapModule         = TestModules;  // NOLINT
using TestMirrorTapUtil           = TestModules;  // NOLINT
using TestModuleRegistry          = TestModules;  // NOLINT
using TestModuleUtil              = TestModules;  // NOLINT
using TestSegmentModules          = TestModules;  // NOLINT
using TestStreamBufferModule      = TestModules;  // NOLINT
using TestSegmentModulesDeathTest = TestModules;  // NOLINT

}  // namespace mrc
