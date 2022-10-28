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

#include "srf/version.hpp"
#include "srf/experimental/modules/segment_module_util.hpp"
#include "srf/experimental/modules/sample_modules.hpp"

#include "test_segment.hpp"

TEST_F(SegmentTests, ModuleRegistryUtilTest)
{
    using namespace modules;

    const auto *registry_namespace = "srf_unittest";

    const std::vector<unsigned int> release_version = {srf_VERSION_MAJOR, srf_VERSION_MINOR, srf_VERSION_PATCH};

    ModelRegistryUtil::register_module<SimpleModule>("SimpleModule", registry_namespace, release_version);
    ModelRegistryUtil::register_module<SourceModule>("SourceModule", registry_namespace, release_version);
    ModelRegistryUtil::register_module<SinkModule>("SinkModule", registry_namespace, release_version);
    ModelRegistryUtil::register_module<NestedModule>("NestedModule", registry_namespace, release_version);
    ModelRegistryUtil::register_module<ConfigurableModule>("ConfigurableModule", registry_namespace, release_version);
    ModelRegistryUtil::register_module<TemplateModule<int>>("TemplateModuleInt", registry_namespace, release_version);
    ModelRegistryUtil::register_module<TemplateModule<std::string>>("TemplateModuleString", registry_namespace, release_version);

    EXPECT_THROW(ModelRegistryUtil::register_module<SimpleModule>("SimpleModule", registry_namespace, release_version),
                                                                  std::invalid_argument);
}
