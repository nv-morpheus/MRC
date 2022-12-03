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

#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/options/options.hpp"

#include <memory>

namespace mrc::internal::system {
class IResources;
}

namespace mrc::internal::executor {

class Executor;

/**
 * @brief The I-classes in mrc/internal enable the building and customization of an Executor.
 *
 * The build order should be:
 * - ISystem
 * - ISystemResources
 * - IExecutor
 *
 * Over time, new customization points will be added between ISystemResources and IExecutor.
 */
class IExecutor
{
  public:
    IExecutor();
    IExecutor(std::shared_ptr<Options>);
    IExecutor(std::unique_ptr<system::IResources>);
    virtual ~IExecutor() = 0;

    void register_pipeline(std::unique_ptr<internal::pipeline::IPipeline> pipeline);

    void start();
    void stop();
    void join();

  protected:
    // this method will be applied

  private:
    std::shared_ptr<Executor> m_impl;
    friend Executor;
};

}  // namespace mrc::internal::executor
