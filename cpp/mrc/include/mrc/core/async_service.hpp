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

#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runnable_resources.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/types.hpp"

#include <chrono>
#include <concepts>
#include <iosfwd>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <utility>
#include <vector>

namespace mrc::runnable {
struct LaunchOptions;
}  // namespace mrc::runnable

namespace mrc {

enum class AsyncServiceState
{
    Initialized,
    Starting,
    Running,
    AwaitingChildren,  // The parent main loop has finished by child ones are still running
    Stopping,
    Killing,
    Completed,
};

/**
 * @brief Converts a `AsyncServiceState` enum to a string
 *
 * @param f
 * @return std::string
 */
inline std::string asyncservicestate_to_str(const AsyncServiceState& s)
{
    switch (s)
    {
    case AsyncServiceState::Initialized:
        return "Initialized";
    case AsyncServiceState::Starting:
        return "Starting";
    case AsyncServiceState::Running:
        return "Running";
    case AsyncServiceState::AwaitingChildren:
        return "AwaitingChildren";
    case AsyncServiceState::Stopping:
        return "Stopping";
    case AsyncServiceState::Killing:
        return "Killing";
    case AsyncServiceState::Completed:
        return "Completed";
    default:
        throw std::logic_error("Unsupported AsyncServiceState enum. Was a new value added recently?");
    }
}

/**
 * @brief Stream operator for `AsyncServiceState`
 *
 * @param os
 * @param f
 * @return std::ostream&
 */
static inline std::ostream& operator<<(std::ostream& os, const AsyncServiceState& f)
{
    os << asyncservicestate_to_str(f);
    return os;
}

class AsyncService : public virtual runnable::IRunnableResourcesProvider
{
  public:
    AsyncService(std::string service_name);
    virtual ~AsyncService();

    const std::string& service_name() const;

    bool is_service_startable() const;
    const AsyncServiceState& state() const;

    SharedFuture<void> service_start();
    void service_await_live();
    void service_stop();
    void service_kill();
    bool service_await_join(std::chrono::milliseconds wait_duration = std::chrono::milliseconds::max());

  protected:
    std::string debug_prefix() const;

    void call_in_destructor();
    void service_set_description(std::string description);
    void mark_started();

    template <typename T>
    requires std::derived_from<T, AsyncService>
    void child_service_start(std::unique_ptr<T>&& child, bool await_live = false)
    {
        return this->child_service_start_impl(std::unique_ptr<AsyncService>(std::move(child)), await_live);
    }

    template <typename T>
    requires std::derived_from<T, AsyncService>
    void child_service_start(std::shared_ptr<T> child, bool await_live = false)
    {
        return this->child_service_start_impl(std::shared_ptr<AsyncService>(std::move(child)), await_live);
    }

    template <typename RunnableT, typename... ContextArgsT>
    void child_runnable_start(std::string name,
                              const mrc::runnable::LaunchOptions& options,
                              std::unique_ptr<RunnableT> runnable,
                              ContextArgsT&&... context_args);

    SharedFuture<void> completed_future() const;

  private:
    void child_service_start_impl(std::shared_ptr<AsyncService> child, bool await_live = false);
    void child_service_start_impl(std::unique_ptr<AsyncService>&& child, bool await_live = false);

    // Advances the state. New state value must be greater than or equal to current state. Using a value less than the
    // current state will generate an error
    bool forward_state(AsyncServiceState new_state, bool assert_forward = false);

    // Ensures the state is at least the current value or higher. Does not change the state if the value is less than or
    // equal the current state
    bool ensure_state(AsyncServiceState ensure_state);

    virtual void do_service_start(std::stop_token stop_token) = 0;
    // virtual void do_service_await_live()                      = 0;
    // virtual void do_service_stop()                            = 0;
    virtual void do_service_kill();
    // virtual void do_service_await_join() = 0;

    AsyncServiceState m_state{AsyncServiceState::Initialized};
    std::string m_service_name{"mrc::AsyncService"};
    SharedPromise<void> m_live_promise;
    SharedFuture<void> m_completed_future;
    bool m_service_await_join_called{false};
    bool m_call_in_destructor_called{false};

    std::stop_source m_stop_source;
    CondVarAny m_cv;
    mutable RecursiveMutex m_mutex;

    std::map<std::string, std::shared_ptr<AsyncService>> m_owned_children;
    std::vector<std::shared_ptr<AsyncService>> m_children;
    // std::vector<SharedFuture<void>> m_child_futures;
};

class AsyncServiceRunnerWrapper : public AsyncService, public runnable::RunnableResourcesProvider
{
  public:
    // AsyncServiceRunnerWrapper(const runnable::RunnableResourcesProvider& resources,
    //                           std::string name,
    //                           std::unique_ptr<runnable::Launcher> launcher) :
    //   AsyncService(name + "(Runnable)"),
    //   runnable::RunnableResourcesProvider(resources),
    //   m_launcher(std::move(launcher))
    // {}

    // AsyncServiceRunnerWrapper(runnable::RunnableResources& resources,
    //                           std::string name,
    //                           std::unique_ptr<runnable::Launcher> launcher) :
    //   AsyncService(name + "(Runnable)"),
    //   runnable::RunnableResourcesProvider(resources),
    //   m_launcher(std::move(launcher))
    // {}

    template <typename RunnableT, typename... ContextArgsT>
    AsyncServiceRunnerWrapper(runnable::IRunnableResourcesProvider& resources,
                              std::string name,
                              const runnable::LaunchOptions& options,
                              std::unique_ptr<RunnableT> runnable,
                              ContextArgsT&&... context_args) :
      AsyncService(name + "(Runnable)"),
      runnable::RunnableResourcesProvider(resources),
      m_launcher(this->runnable().launch_control().prepare_launcher(options,
                                                                    std::move(runnable),
                                                                    std::forward<ContextArgsT>(context_args)...))
    {}

    // template <typename RunnableT, typename... ContextArgsT>
    // [[nodiscard]] static std::unique_ptr<AsyncServiceRunnerWrapper> create(
    //     runnable::IRunnableResourcesProvider& resources,
    //     std::string name,
    //     const runnable::LaunchOptions& options,
    //     std::unique_ptr<RunnableT> runnable,
    //     ContextArgsT&&... context_args)
    // {
    //     auto launcher = resources.runnable().launch_control().prepare_launcher(
    //         options,
    //         std::move(runnable),
    //         std::forward<ContextArgsT>(context_args)...);

    //     auto a = std::make_unique<AsyncServiceRunnerWrapper>(resources, std::move(name), std::move(launcher));

    //     a->runnable().launch_control();

    //     return a;
    // }

  private:
    void do_service_start(std::stop_token stop_token) final
    {
        {
            // Prevent changes while we are using m_runner
            std::lock_guard<decltype(m_runner_mutex)> lock(m_runner_mutex);

            m_runner = m_launcher->ignition();
        }

        std::stop_callback stop_callback(stop_token, [this]() {
            // Prevent changes while we are using m_runner
            std::lock_guard<decltype(m_runner_mutex)> lock(m_runner_mutex);

            m_runner->stop();
        });

        this->mark_started();

        m_runner->await_join();
    }
    void do_service_kill() final
    {
        // Prevent changes while we are using m_runner
        std::lock_guard<decltype(m_runner_mutex)> lock(m_runner_mutex);

        if (m_runner)
        {
            m_runner->kill();
        }
    }

    mutable RecursiveMutex m_runner_mutex;

    std::unique_ptr<runnable::Launcher> m_launcher;
    std::unique_ptr<runnable::Runner> m_runner;
};

template <typename RunnableT, typename... ContextArgsT>
void AsyncService::child_runnable_start(std::string name,
                                        const mrc::runnable::LaunchOptions& options,
                                        std::unique_ptr<RunnableT> runnable,
                                        ContextArgsT&&... context_args)
{
    auto wrapper = std::make_unique<AsyncServiceRunnerWrapper>(*this,
                                                               std::move(name),
                                                               options,
                                                               std::move(runnable),
                                                               std::forward<ContextArgsT>(context_args)...);

    this->child_service_start(std::move(wrapper));
}

}  // namespace mrc
