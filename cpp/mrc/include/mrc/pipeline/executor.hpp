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

#pragma once

#include "mrc/utils/macros.hpp"

#include <memory>

namespace mrc {
class Options;
}  // namespace mrc
namespace mrc::pipeline {
class IPipeline;
}  // namespace mrc::pipeline

namespace mrc::pipeline {
class ISystem;

class IExecutor
{
  public:
    virtual ~IExecutor() = default;

    DELETE_COPYABILITY(IExecutor);

    virtual void register_pipeline(std::shared_ptr<IPipeline> pipeline) = 0;
    virtual void start()                                                = 0;
    virtual void stop()                                                 = 0;
    virtual void join()                                                 = 0;

  protected:
    IExecutor() = default;
};

}  // namespace mrc::pipeline

namespace mrc {

// For backwards compatibility, make utility implementation which holds onto a unique_ptr
class Executor : public pipeline::IExecutor
{
  public:
    Executor();
    Executor(std::shared_ptr<Options> options);
    ~Executor() override;

    void register_pipeline(std::shared_ptr<pipeline::IPipeline> pipeline) override;
    void start() override;
    void stop() override;
    void join() override;

  private:
    std::unique_ptr<IExecutor> m_impl;
};

std::unique_ptr<pipeline::IExecutor> make_executor(std::shared_ptr<Options> options);

std::unique_ptr<pipeline::IExecutor> make_executor(std::unique_ptr<pipeline::ISystem> system);

}  // namespace mrc
