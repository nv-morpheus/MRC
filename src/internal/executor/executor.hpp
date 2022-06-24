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
#include "internal/resources/forward.hpp"
#include "internal/service.hpp"
#include "internal/system/forward.hpp"

#include "srf/engine/pipeline/ipipeline.hpp"
#include "srf/options/options.hpp"
#include "srf/types.hpp"

#include <memory>
#include <utility>

namespace srf::internal::executor {

/**
 * @brief Common Executor code used by both the Standalone and Architect Executors
 *
 * Issues #149 will begin to separate some of the functionality of ExeuctorBase into individual components.
 */
class Executor : public Service
{
  public:
    Executor(Handle<Options> options);
    Executor(Handle<system::System> system);
    ~Executor() override;

    void register_pipeline(std::unique_ptr<pipeline::IPipeline> ipipeline);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    Handle<system::System> m_system;
    Handle<resources::ResourcePartitions> m_resources;

    std::unique_ptr<pipeline::Manager> m_pipeline_manager;
};

inline std::unique_ptr<Executor> make_executor(Handle<Options> options)
{
    return std::make_unique<Executor>(std::move(options));
}

inline std::unique_ptr<Executor> make_executor(Handle<system::System> sytem)
{
    return std::make_unique<Executor>(std::move(sytem));
}

}  // namespace srf::internal::executor
