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

#include "internal/async_service.hpp"

#include "mrc/channel/status.hpp"

#include <glog/logging.h>

#include <chrono>
#include <mutex>
#include <ostream>
#include <stop_token>
#include <utility>

namespace mrc {

AsyncService::AsyncService(std::string service_name) : m_service_name(std::move(service_name)) {}

AsyncService::~AsyncService()
{
    auto state = this->state();

    bool is_running = state > AsyncServiceState::Initialized && state < AsyncServiceState::Completed;

    CHECK(!is_running) << "Must call AsyncService::call_in_destructor to ensure service is cleaned up before being "
                          "destroyed";
}

bool AsyncService::is_service_startable() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return (m_state == AsyncServiceState::Initialized);
}

const AsyncServiceState& AsyncService::state() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_state;
}

Future<void> AsyncService::service_start(std::stop_source stop_source)
{
    // Lock here since we do stuff after checking the state which should be synced
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // This throws if we are already started
    forward_state(AsyncServiceState::Starting, true);

    // Copy the stop_source state (either it was passed in or created)
    m_stop_source = stop_source;

    return this->runnable().main().enqueue([this]() {
        // Add a stop callback to notify the cv anytime a stop is requested
        std::stop_callback stop_callback(m_stop_source.get_token(), [this]() {
            m_cv.notify_all();
        });

        do_service_start(m_stop_source.get_token());

        {
            // Get the mutex to prevent changes to m_child_futures
            std::unique_lock<decltype(m_mutex)> lock(m_mutex);

            CHECK(m_state >= AsyncServiceState::Running) << this->debug_prefix()
                                                         << " did not start up properly. Must call mark_started() "
                                                            "inside "
                                                            "of do_service_start()";

            // Set the state to stopping to prevent any changes while we wait for children. Dont check the response
            // since it could have been requested by external users
            forward_state(AsyncServiceState::Stopping);
        }

        // Wait for all children to have completed. Make sure not to hold the lock when waiting on children.
        // m_child_futures cant be changed after the state is set to stopping
        for (auto& f : m_child_futures)
        {
            f.wait();
        }

        DCHECK(forward_state(AsyncServiceState::Completed));
    });
}

void AsyncService::service_await_live()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // DVLOG(20) << this->debug_prefix() << " entering service_await_live(). State: " << m_state;

    m_cv.wait(lock, [this]() {
        // DVLOG(20) << this->debug_prefix() << " checking service_await_live(). State: " << m_state;
        return m_state >= AsyncServiceState::Running || m_stop_source.stop_requested();
    });

    // DVLOG(20) << this->debug_prefix() << " leaving service_await_live(). State: " << m_state;
}

void AsyncService::service_stop()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // Ensures this only gets executed once
    if (forward_state(AsyncServiceState::Stopping))
    {
        DCHECK(m_stop_source.stop_possible()) << this->debug_prefix() << " Invalid state. Cannot request a stop";

        m_stop_source.request_stop();
    }
}

void AsyncService::service_kill()
{
    // First, make sure we have tried to stop first. Should be a no-op if already stopped
    this->service_stop();

    // Ensures this only gets executed once
    if (forward_state(AsyncServiceState::Killing))
    {
        // Kill all children first
        for (auto& child : m_children)
        {
            child.get().service_kill();
        }

        // Attempt the service specific kill operation
        do_service_kill();
    }
}

void AsyncService::service_await_join()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    m_cv.wait(lock, [this]() {
        return m_state >= AsyncServiceState::Completed;
    });
}

std::string AsyncService::debug_prefix() const
{
    return MRC_CONCAT_STR("Service[" << m_service_name << "]:");
}

void AsyncService::call_in_destructor()
{
    auto state = this->state();
    if (state > AsyncServiceState::Initialized)
    {
        if (state == AsyncServiceState::Running)
        {
            LOG(ERROR) << this->debug_prefix()
                       << " service was not stopped/killed before being destructed; issuing kill";
            service_kill();
        }

        if (state != AsyncServiceState::Completed)
        {
            LOG(ERROR) << this->debug_prefix() << " service was not joined before being destructed; issuing join";
            service_await_join();
        }
    }
}

void AsyncService::service_set_description(std::string description)
{
    m_service_name = std::move(description);
}

void AsyncService::mark_started()
{
    decltype(m_children) children;
    {
        // Lock to prevent changes to the state and children
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);

        // Copy the children outside of the lock
        children = m_children;
    }

    // Before indicating that we are running, we need any children added during startup to be marked as ready
    // DO NOT HOLD THE LOCK HERE!
    for (auto& child : children)
    {
        child.get().service_await_live();
    }

    forward_state(AsyncServiceState::Running, true);
}

void AsyncService::child_service_start(AsyncService& child)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // The state must be running
    CHECK(m_state == AsyncServiceState::Starting || m_state == AsyncServiceState::Running) << "Can only start child "
                                                                                              "service in Starting or "
                                                                                              "Running state";

    m_children.emplace_back(child);
    m_child_futures.emplace_back(child.service_start(m_stop_source));
}

bool AsyncService::forward_state(AsyncServiceState new_state, bool assert_forward)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    if (assert_forward)
    {
        CHECK(m_state <= new_state) << this->debug_prefix()
                                    << " invalid AsyncServiceState requested; AsyncServiceState is only allowed to "
                                       "advance";
    }

    if (m_state < new_state)
    {
        DVLOG(20) << this->debug_prefix() << " changing state. From: " << m_state << " to " << new_state;

        m_state = new_state;

        // Notify the CV for anyone waiting on this service
        m_cv.notify_all();

        return true;
    }
    return false;
}

void AsyncService::do_service_kill()
{
    // Nothing in base
}

}  // namespace mrc
