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

#include "srf/experimental/modules/module_registry_util.hpp"
#include "srf/experimental/modules/segment_modules.hpp"
#include "srf/segment/builder.hpp"
#include "srf/version.hpp"

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

const std::vector<unsigned int> DynamicTestModuleVersion{srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

const char* MODULES[] = {"srf_unittest_cpp_dynamic::DynamicSourceModule",
                         "srf_unittest_cpp_dynamic_2::DynamicSourceModule",
                         "srf_unittest_cpp_dynamic_3::DynamicSourceModule"};

[[maybe_unused]] bool SRF_MODULE_dummy_entrypoint()  // NOLINT
{
    return true;
}

[[maybe_unused]] bool SRF_MODULE_entrypoint_load()  // NOLINT
{
    using namespace srf::modules;

    try
    {
        ModuleRegistry::register_module("DynamicSourceModule",
                                        "srf_unittest_cpp_dynamic",
                                        DynamicTestModuleVersion,
                                        [](std::string module_name, nlohmann::json config) {
                                            return std::make_shared<srf::modules::DynamicSourceModule>(
                                                std::move(module_name), std::move(config));
                                        });

        ModuleRegistry::register_module("DynamicSourceModule",
                                        "srf_unittest_cpp_dynamic_2",
                                        DynamicTestModuleVersion,
                                        [](std::string module_name, nlohmann::json config) {
                                            return std::make_shared<srf::modules::DynamicSourceModule>(
                                                std::move(module_name), std::move(config));
                                        });

        ModuleRegistry::register_module("DynamicSourceModule",
                                        "srf_unittest_cpp_dynamic_3",
                                        DynamicTestModuleVersion,
                                        [](std::string module_name, nlohmann::json config) {
                                            return std::make_shared<srf::modules::DynamicSourceModule>(
                                                std::move(module_name), std::move(config));
                                        });
    } catch (...)
    {
        return false;
    }

    return true;
}

[[maybe_unused]] bool SRF_MODULE_entrypoint_unload()  // NOLINT
{
    using namespace srf::modules;

    try
    {
        ModuleRegistry::unregister_module("DynamicSourceModule", "srf_unittest_cpp_dynamic");
        ModuleRegistry::unregister_module("DynamicSourceModule", "srf_unittest_cpp_dynamic_2");
        ModuleRegistry::unregister_module("DynamicSourceModule", "srf_unittest_cpp_dynamic_3");
    } catch (...)
    {
        return false;
    }

    return true;
}

[[maybe_unused]] unsigned int SRF_MODULE_entrypoint_list(const char** result)  // NOLINT
{
    *result = (const char*)(&MODULES);

    return 3;
}

[[maybe_unused]] bool SRF_MODULE_bad_version_entrypoint()  // NOLINT
{
    using namespace srf::modules;

    auto BadVersion = std::vector<unsigned int>{13, 14, 15};

    ModuleRegistry::register_module("DynamicSourceModule_BAD",
                                    "srf_unittest_cpp_dynamic_BAD",
                                    BadVersion,
                                    [](std::string module_name, nlohmann::json config) {
                                        return std::make_shared<srf::modules::DynamicSourceModule>(
                                            std::move(module_name), std::move(config));
                                    });

    return true;
}
}
