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

#include "srf/experimental/modules/segment_module_util.hpp"
#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/builder.hpp"

#include <nlohmann/json.hpp>
#include <rxcpp/rx-subscriber.hpp>

namespace srf::modules {
class DynamicSourceModule : public SegmentModule
{
  public:
    DynamicSourceModule(std::string module_name);
    DynamicSourceModule(std::string module_name, nlohmann::json config);

    bool m_was_configured{false};

  protected:
    void initialize(segment::Builder& builder) override;

  private:
    bool m_initialized{false};
};

DynamicSourceModule::DynamicSourceModule(std::string module_name) : SegmentModule(std::move(module_name)) {}
DynamicSourceModule::DynamicSourceModule(std::string module_name, nlohmann::json config) :
  SegmentModule(std::move(module_name), std::move(config))
{}

void DynamicSourceModule::initialize(segment::Builder& builder)
{
    unsigned int count{1};

    if (config().contains("source_count"))
    {
        count = config()["source_count"];
    }

    auto source = builder.make_source<bool>("source", [count](rxcpp::subscriber<bool>& sub) {
        if (sub.is_subscribed())
        {
            for (unsigned int i = 0; i < count; ++i)
            {
                sub.on_next(true);
            }
        }

        sub.on_completed();
    });

    // Register the submodules output as one of this module's outputs
    register_output_port("source", source, source->object().source_type());
}

}  // namespace srf::modules

extern "C" {
[[maybe_unused]] bool SRF_MODULE_dummy_entrypoint()
{
    // std::cerr << "Initializing dynamic module test library" << std::endl;
    return true;
}

[[maybe_unused]] bool SRF_MODULE_entrypoint()
{
    using namespace srf::modules;

    try
    {
        ModuleRegistry::register_module(
            "DynamicSourceModule",
            [](std::string module_name, nlohmann::json config) {
                return std::make_shared<srf::modules::DynamicSourceModule>(std::move(module_name), std::move(config));
            },
            "srf_unittest_cpp_dynamic");

        ModuleRegistry::register_module(
            "DynamicSourceModule",
            [](std::string module_name, nlohmann::json config) {
                return std::make_shared<srf::modules::DynamicSourceModule>(std::move(module_name), std::move(config));
            },
            "srf_unittest_cpp_dynamic_2");

        ModuleRegistry::register_module(
            "DynamicSourceModule",
            [](std::string module_name, nlohmann::json config) {
                return std::make_shared<srf::modules::DynamicSourceModule>(std::move(module_name), std::move(config));
            },
            "srf_unittest_cpp_dynamic_3");
    } catch (...)
    {
        return false;
    }

    return true;
}
}
