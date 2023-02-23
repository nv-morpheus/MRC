/**
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

#include "internal/control_plane/server/connection_manager.hpp"
#include "internal/service.hpp"

#include "mrc/core/error.hpp"

#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/mutex.hpp>
#include <rxcpp/rx.hpp>

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>

namespace node {
class Environment;
class CommonEnvironmentSetup;
class InitializationResult;
class MultiIsolatePlatform;
}  // namespace node

namespace mrc::internal::control_plane {

class NodeContext : public ::mrc::runnable::Context
{
  public:
  protected:
    void do_init() override;

  private:
    void launch_node(std::vector<std::string> args);

    std::unique_ptr<::node::InitializationResult> m_init_result;
    std::unique_ptr<::node::MultiIsolatePlatform> m_platform;

    std::unique_ptr<::node::CommonEnvironmentSetup> m_setup;
};

class NodeRuntime : public ::mrc::runnable::RunnableWithContext<::mrc::runnable::Context>
{
  public:
    NodeRuntime(std::vector<std::string> args);
    ~NodeRuntime() override;

    void start();
    void stop();
    void kill();

  private:
    void run(::mrc::runnable::Context& ctx) override;
    void on_state_update(const Runnable::State& state) override;

    // void run_node();

    // std::unique_ptr<::node::CommonEnvironmentSetup> node_init_setup(std::vector<std::string> args);
    // void node_run_environment(std::unique_ptr<::node::CommonEnvironmentSetup>);
    void launch_node(std::vector<std::string> args);

    std::unique_ptr<::node::InitializationResult> m_init_result;
    std::unique_ptr<::node::MultiIsolatePlatform> m_platform;

    std::unique_ptr<::node::CommonEnvironmentSetup> m_setup;

    std::vector<std::string> m_args;
};

class NodeService : public Service
{
  public:
    NodeService(runnable::RunnableResources& runnable);
    ~NodeService() override;

    void set_args(std::vector<std::string> args);

  private:
    void do_service_start() final;
    void do_service_stop() final;
    void do_service_kill() final;
    void do_service_await_live() final;
    void do_service_await_join() final;

    void launch_node(std::vector<std::string> args);

    // mrc resources
    runnable::RunnableResources& m_runnable;

    std::unique_ptr<::node::InitializationResult> m_init_result;
    std::unique_ptr<::node::MultiIsolatePlatform> m_platform;

    std::unique_ptr<::node::CommonEnvironmentSetup> m_setup;

    std::vector<std::string> m_args;
    bool m_launch_node{true};

    mutable boost::fibers::mutex m_mutex;

    std::thread m_node_thread;
    Promise<void> m_started_promise{};
    Future<void> m_started_future;
    Future<void> m_completed_future;
};
}  // namespace mrc::internal::control_plane
