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

#include "test_modules.hpp"

#include "mrc/modules/module_registry_util.hpp"
#include "mrc/modules/sample_modules.hpp"
#include "mrc/version.hpp"

TEST_F(TestModuleUtil, ModuleRegistryUtilTest)
{
    using namespace modules;

    const auto* registry_namespace = "mrc_unittest";

    const std::vector<unsigned int> release_version = {mrc_VERSION_MAJOR, mrc_VERSION_MINOR, mrc_VERSION_PATCH};

    ModelRegistryUtil::create_registered_module<SimpleModule>("SimpleModule", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<SourceModule>("SourceModule", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<SinkModule>("SinkModule", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<NestedModule>("NestedModule", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<ConfigurableModule>(
        "ConfigurableModule", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<TemplateModule<int>>(
        "TemplateModuleInt", registry_namespace, release_version);
    ModelRegistryUtil::create_registered_module<TemplateModule<std::string>>(
        "TemplateModuleString", registry_namespace, release_version);

    EXPECT_THROW(
        ModelRegistryUtil::create_registered_module<SimpleModule>("SimpleModule", registry_namespace, release_version),
        std::invalid_argument);
}
