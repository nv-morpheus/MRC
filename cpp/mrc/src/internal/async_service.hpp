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

#include "internal/runnable/resources.hpp"

#include "mrc/types.hpp"

#include <functional>
#include <mutex>
#include <stop_token>
#include <string>

namespace mrc::internal {

enum class AsyncServiceState
{
    Initialized,
    Starting,
    Running,
    Stopping,
    Killing,
    Completed,
};

class AsyncService : public virtual runnable::IRunnableResourcesProvider
{
  public:
    AsyncService();
    virtual ~AsyncService();

    bool is_service_startable() const;
    const AsyncServiceState& state() const;

    Future<void> service_start(std::stop_source stop_source = {});
    void service_await_live();
    void service_stop();
    void service_kill();
    void service_await_join();

  protected:
    void call_in_destructor();
    void service_set_description(std::string description);
    void mark_started();

    void child_service_start(AsyncService& child);

  private:
    bool forward_state(AsyncServiceState new_state, bool assert_forward = false);

    virtual void do_service_start(std::stop_token stop_token) = 0;
    // virtual void do_service_await_live()                      = 0;
    // virtual void do_service_stop()                            = 0;
    virtual void do_service_kill();
    // virtual void do_service_await_join() = 0;

    AsyncServiceState m_state{AsyncServiceState::Initialized};
    std::string m_description{"mrc::internal::service"};

    std::stop_source m_stop_source;
    CondVarAny m_cv;
    mutable RecursiveMutex m_mutex;

    std::vector<std::reference_wrapper<AsyncService>> m_children;
    std::vector<Future<void>> m_child_futures;
};

}  // namespace mrc::internal
