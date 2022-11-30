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

#include "internal/pipeline/forward.hpp"
#include "internal/resources/manager.hpp"
#include "internal/service.hpp"
#include "internal/system/resources.hpp"
#include "internal/system/system_provider.hpp"

#include "mrc/engine/pipeline/ipipeline.hpp"
#include "mrc/options/options.hpp"

#include <memory>
#include <utility>

namespace mrc::internal::executor {

/**
 * @brief Common Executor code used by both the Standalone and Architect Executors
 *
 * Issues #149 will begin to separate some of the functionality of ExeuctorBase into individual components.
 */
class Executor : public Service, public system::SystemProvider
{
  public:
    Executor(std::shared_ptr<Options> options);
    Executor(std::unique_ptr<system::Resources> resources);
    ~Executor() override;

    void register_pipeline(std::unique_ptr<pipeline::IPipeline> ipipeline);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    std::unique_ptr<resources::Manager> m_resources_manager;
    std::unique_ptr<pipeline::Manager> m_pipeline_manager;
};

inline std::unique_ptr<Executor> make_executor(std::shared_ptr<Options> options)
{
    return std::make_unique<Executor>(std::move(options));
}

inline std::unique_ptr<Executor> make_executor(std::unique_ptr<system::Resources> resources)
{
    return std::make_unique<Executor>(std::move(resources));
}

}  // namespace mrc::internal::executor
