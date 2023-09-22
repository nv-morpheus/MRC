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

#include "internal/service.hpp"

#include "mrc/core/utils.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>

#include <exception>
#include <functional>  // for function
#include <mutex>
#include <sstream>  // for operator<<, basic_ostream
#include <utility>

namespace mrc {

Service::Service(std::string service_name) : m_service_name(std::move(service_name)) {}

Service::~Service()
{
    if (!m_call_in_destructor_called)
    {
        LOG(ERROR) << "Must call Service::call_in_destructor to ensure service is cleaned up before being "
                      "destroyed";
    }

    auto state = this->state();
    CHECK(state == ServiceState::Initialized || state == ServiceState::Completed);
}

const std::string& Service::service_name() const
{
    return m_service_name;
}

bool Service::is_service_startable() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return (m_state == ServiceState::Initialized);
}

bool Service::is_running() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return (m_state > ServiceState::Initialized && m_state < ServiceState::Completed);
}

const ServiceState& Service::state() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    return m_state;
}

void Service::service_start()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    if (!this->is_service_startable())
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR(this->debug_prefix() << " Service has already been started"));
    }

    if (advance_state(ServiceState::Starting))
    {
        // Unlock the mutex before calling start to avoid a deadlock
        lock.unlock();

        try
        {
            this->do_service_start();

            // Use ensure_state here in case the service itself called stop or kill
            this->desired_state(ServiceState::Running);
        } catch (...)
        {
            // On error, set this to completed and rethrow the error to allow for cleanup
            this->advance_state(ServiceState::Completed);

            throw;
        }
    }
}

void Service::service_await_live()
{
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        if (this->is_service_startable())
        {
            throw exceptions::MrcRuntimeError(MRC_CONCAT_STR(this->debug_prefix() << " Service must be started before "
                                                                                     "awaiting live"));
        }

        // Check if this is our first call to service_await_join
        if (!m_service_await_live_called)
        {
            // Prevent reentry
            m_service_await_live_called = true;

            // We now create a promise and a future to track the completion of this function
            Promise<void> live_promise;

            m_live_future = live_promise.get_future();

            // Unlock the mutex before calling await to avoid a deadlock
            lock.unlock();

            try
            {
                // Now call the await join (this can throw!)
                this->do_service_await_live();

                // Set the value only if there was not an exception
                live_promise.set_value();

            } catch (...)
            {
                // Join must have thrown, set the exception in the promise (it will be retrieved later)
                live_promise.set_exception(std::current_exception());
            }
        }
    }

    // Wait for the future to be returned. This will rethrow any exception thrown in do_service_await_join
    m_live_future.get();
}

void Service::service_stop()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    if (this->is_service_startable())
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR(this->debug_prefix() << " Service must be started before "
                                                                                 "stopping"));
    }

    // Ensure we are at least in the stopping state. If so, execute the stop call
    if (this->desired_state(ServiceState::Stopping))
    {
        lock.unlock();

        this->do_service_stop();
    }
}

void Service::service_kill()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    if (this->is_service_startable())
    {
        throw exceptions::MrcRuntimeError(MRC_CONCAT_STR(this->debug_prefix() << " Service must be started before "
                                                                                 "killing"));
    }

    // Ensure we are at least in the stopping state. If so, execute the stop call
    if (this->desired_state(ServiceState::Killing))
    {
        lock.unlock();

        this->do_service_kill();
    }
}

void Service::service_await_join()
{
    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        if (this->is_service_startable())
        {
            throw exceptions::MrcRuntimeError(MRC_CONCAT_STR(this->debug_prefix() << " Service must be started before "
                                                                                     "awaiting join"));
        }

        // Check if this is our first call to service_await_join
        if (!m_service_await_join_called)
        {
            // Prevent reentry
            m_service_await_join_called = true;

            // We now create a promise and a future to track the completion of the service
            Promise<void> completed_promise;

            m_completed_future = completed_promise.get_future();

            // Unlock the mutex before calling await join to avoid a deadlock
            lock.unlock();

            try
            {
                {
                    Unwinder ensure_completed_set([this]() {
                        // Always set the state to completed before releasing the future
                        this->advance_state(ServiceState::Completed);
                    });

                    // Now call the await join (this can throw!)
                    this->do_service_await_join();
                }

                // Set the value only if there was not an exception
                completed_promise.set_value();

            } catch (const std::exception& ex)
            {
                LOG(ERROR) << this->debug_prefix() << " caught exception in service_await_join: " << ex.what();
                // Join must have thrown, set the exception in the promise (it will be retrieved later)
                completed_promise.set_exception(std::current_exception());
            }
        }
    }

    // Wait for the completed future to be returned. This will rethrow and exception thrown in do_service_await_join
    m_completed_future.get();
}

std::string Service::debug_prefix() const
{
    return MRC_CONCAT_STR("Service[" << m_service_name << "]:");
}

void Service::call_in_destructor()
{
    // Guarantee that we set the flag that this was called
    Unwinder ensure_flag([this]() {
        m_call_in_destructor_called = true;
    });

    auto state = this->state();
    if (state > ServiceState::Initialized)
    {
        if (state == ServiceState::Running)
        {
            LOG(ERROR) << this->debug_prefix()
                       << ": service was not stopped/killed before being destructed; issuing kill";
            this->service_kill();
        }

        if (state != ServiceState::Completed)
        {
            LOG(ERROR) << this->debug_prefix() << ": service was not joined before being destructed; issuing join";
            this->service_await_join();
        }
    }
}

void Service::service_set_description(std::string description)
{
    m_service_name = std::move(description);
}

bool Service::advance_state(ServiceState new_state, bool assert_state_change)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // State needs to always be moving foward or the same
    CHECK_GE(new_state, m_state) << this->debug_prefix()
                                 << " invalid ServiceState requested; ServiceState is only allowed to advance. "
                                    "Current: "
                                 << m_state << ", Requested: " << new_state;

    if (m_state < new_state)
    {
        DVLOG(20) << this->debug_prefix() << " advancing state. From: " << m_state << " to " << new_state;

        m_state = new_state;

        return true;
    }

    CHECK(!assert_state_change) << this->debug_prefix()
                                << " invalid ServiceState requested; ServiceState was required to move forward "
                                   "but the state was already set to "
                                << m_state;

    return false;
}

bool Service::desired_state(ServiceState ensure_state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    if (ensure_state > m_state)
    {
        return advance_state(ensure_state);
    }

    return false;
}

}  // namespace mrc
