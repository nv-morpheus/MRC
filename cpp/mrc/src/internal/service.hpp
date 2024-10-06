/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/types.hpp"

#include <iosfwd>     // for ostream
#include <stdexcept>  // for logic_error
#include <string>

namespace mrc {

enum class ServiceState
{
    Initialized,
    Starting,
    Running,
    Stopping,
    Killing,
    Completed,
};

/**
 * @brief Converts a `ServiceState` enum to a string
 *
 * @param f
 * @return std::string
 */
inline std::string servicestate_to_str(const ServiceState& s)
{
    switch (s)
    {
    case ServiceState::Initialized:
        return "Initialized";
    case ServiceState::Starting:
        return "Starting";
    case ServiceState::Running:
        return "Running";
    case ServiceState::Stopping:
        return "Stopping";
    case ServiceState::Killing:
        return "Killing";
    case ServiceState::Completed:
        return "Completed";
    default:
        throw std::logic_error("Unsupported ServiceState enum. Was a new value added recently?");
    }
}

/**
 * @brief Stream operator for `AsyncServiceState`
 *
 * @param os
 * @param f
 * @return std::ostream&
 */
static inline std::ostream& operator<<(std::ostream& os, const ServiceState& f)
{
    os << servicestate_to_str(f);
    return os;
}

class Service
{
  public:
    virtual ~Service();

    const std::string& service_name() const;

    bool is_service_startable() const;

    bool is_running() const;

    const ServiceState& state() const;

    void service_start();
    void service_await_live();
    void service_stop();
    void service_kill();
    void service_await_join();

  protected:
    Service(std::string service_name);

    // Prefix to use for debug messages. Contains useful information about the service
    std::string debug_prefix() const;

    void call_in_destructor();
    void service_set_description(std::string description);

  private:
    // Advances the state. New state value must be greater than or equal to current state. Using a value less than the
    // current state will generate an error. Use assert_forward = false to require that the state advances. Normally,
    // same states are fine
    bool advance_state(ServiceState new_state, bool assert_state_change = false);

    // Ensures the state is at least the current value or higher. Does not change the state if the value is less than or
    // equal the current state
    bool ensure_state(ServiceState desired_state);

    virtual void do_service_start()      = 0;
    virtual void do_service_await_live() = 0;
    virtual void do_service_stop()       = 0;
    virtual void do_service_kill()       = 0;
    virtual void do_service_await_join() = 0;

    ServiceState m_state{ServiceState::Initialized};
    std::string m_service_name{"mrc::Service"};

    // This future is set in `service_await_live` and is used to wait for the service to to be live. We use a future
    // here in case it is called multiple times, so that all callers will all be released when the service is live.
    SharedFuture<void> m_live_future;

    // This future is set in `service_await_join` and is used to wait for the service to complete. We use a future here
    // in case join is called multiple times, so that all callers will all be released when the service completes.
    SharedFuture<void> m_completed_future;

    bool m_service_await_live_called{false};
    bool m_service_await_join_called{false};
    bool m_call_in_destructor_called{false};

    mutable RecursiveMutex m_mutex;
};

}  // namespace mrc
