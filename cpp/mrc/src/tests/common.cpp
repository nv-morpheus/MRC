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

#include "common.hpp"

#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/threading_resources.hpp"

#include "mrc/options/options.hpp"

#include <memory>
#include <utility>

namespace mrc::tests {

std::unique_ptr<system::SystemDefinition> make_system(std::function<void(Options&)> updater)
{
    auto options = std::make_shared<Options>();
    if (updater)
    {
        updater(*options);
    }

    return std::make_unique<system::SystemDefinition>(std::move(options));
}

std::unique_ptr<system::ThreadingResources> make_threading_resources(std::function<void(Options&)> updater)
{
    auto system = make_system(updater);

    return make_threading_resources(std::move(system));
}

std::unique_ptr<system::ThreadingResources> make_threading_resources(std::unique_ptr<system::SystemDefinition> system)
{
    return std::make_unique<system::ThreadingResources>(system::SystemProvider(std::move(system)));
}

}  // namespace mrc::tests
