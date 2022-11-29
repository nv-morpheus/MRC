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

#include "srf/runnable/engine.hpp"
#include "srf/runnable/types.hpp"

#include <memory>

namespace srf::runnable {

/**
 * @brief Constructs Engines of a specified type
 *
 * EngineFactory are specific to EngineType which it can build. These factories hold the resources required to build
 * engines of a given type. SingleUse engine factories will throw an error if factory does not have enough resources to
 * build the number of requested engines. Reusable factories should not error on resources as each task queue (fibers)
 * or cpu_id is reusable. Reusable factories oversubscribe threads with fibers and cores with threads.
 *
 * FiberEngineFactory and ThreadEngineFactory are the two specializations.
 */
struct EngineFactory
{
    virtual ~EngineFactory()           = default;
    virtual EngineType backend() const = 0;  // todo(cpp20) constexpr virtual for the specific types
    virtual std::shared_ptr<Engines> build_engines(const LaunchOptions& launch_options) = 0;
};

}  // namespace srf::runnable
