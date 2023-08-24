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

#include "mrc/core/async_service.hpp"

#include "mrc/core/task_queue.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <glog/logging.h>

#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <ratio>
#include <sstream>
#include <stop_token>
#include <utility>

namespace mrc {

AsyncService::AsyncService(std::string service_name) : m_service_name(std::move(service_name)) {}

AsyncService::~AsyncService()
{
    if (!m_call_in_destructor_called)
    {
        LOG(ERROR) << "Must call AsyncService::call_in_destructor to ensure service is cleaned up before being "
                      "destroyed";
    }
}

const std::string& AsyncService::service_name() const
{
    return m_service_name;
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

SharedFuture<void> AsyncService::service_start()
{
    // Lock here since we do stuff after checking the state which should be synced
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // This throws if we are already started
    this->forward_state(AsyncServiceState::Starting, true);

    // // Copy the stop_source state (either it was passed in or created)
    // m_stop_source = stop_source;

    // Save to a shared future so we can also check the return in join()
    m_completed_future = this->runnable().main().enqueue([this]() {
        Unwinder ensure_completed([this]() {
            // Ensure that the state gets set to completed even if we error
            LOG_IF(ERROR, !this->forward_state(AsyncServiceState::Completed))
                << this->debug_prefix() << " Inconsistent state. Could not set to Completed";
        });

        // // Add a stop callback to notify the cv anytime a stop is requested
        // std::stop_callback stop_callback(m_stop_source.get_token(), [this]() {
        //     std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        //     // Ensure that our state is set to stopping
        //     this->ensure_state(AsyncServiceState::Stopping);

        //     m_cv.notify_all();
        // });

        try
        {
            do_service_start(m_stop_source.get_token());
        } catch (const std::exception& ex)
        {
            AsyncServiceState current_state;

            {
                // Only temporarily hold the lock here to get the current state and prevent changes to the children
                std::unique_lock<decltype(m_mutex)> lock(m_mutex);

                // Save the state so we can release the lock
                current_state = m_state;

                // Set the state to awaiting children. This prevents any changes to m_child_futures
                this->ensure_state(AsyncServiceState::AwaitingChildren);
            }

            // Log the error
            LOG(ERROR) << this->debug_prefix() << " error occurred in service. Error message: " << ex.what();

            // Give the children a chance to stop gracefully
            for (auto& child : m_children)
            {
                child->service_stop();
            }

            // Wait a small amount of time for the children to stop gracefully
            for (auto& child : m_children)
            {
                auto child_future = child->completed_future();

                // Only wait() here. Dont care about child errors since we already have an exception
                if (child_future.wait_for(std::chrono::milliseconds(100)) == boost::fibers::future_status::timeout)
                {
                    // Didnt stop in time. Kill the service
                    child->service_kill();

                    // Try and wait one more time (just in case)
                    if (child_future.wait_for(std::chrono::milliseconds(100)) == boost::fibers::future_status::timeout)
                    {
                        LOG(ERROR) << this->debug_prefix()
                                   << " could not shut down child in 200 ms. Child: " << child->debug_prefix();
                    }
                }
            }

            // Set the error in the live exception if we didnt fully start
            if (current_state < AsyncServiceState::Running)
            {
                m_live_promise.set_exception(std::current_exception());
            }

            // Rethrow to set the exception in the completed future
            throw;
        }

        // Stopped without any errors.
        {
            // Get the mutex to prevent changes to m_child_futures. Must release the lock before working with m_children
            std::unique_lock<decltype(m_mutex)> lock(m_mutex);

            CHECK(m_state >= AsyncServiceState::Running) << this->debug_prefix()
                                                         << " did not start up properly. Must call mark_started() "
                                                            "inside of do_service_start()";

            // Set the state to awaiting children to prevent any changes while we wait for children. Only use
            // ensure_state here in case we were stopped or killed
            this->ensure_state(AsyncServiceState::AwaitingChildren);
        }

        // Wait for all children to have completed (only wait() to make sure they all stopped before pulling any
        // exceptions)
        for (auto& child : m_children)
        {
            child->completed_future().wait();
        }

        // Now join them to pull any exceptions
        for (auto& child : m_children)
        {
            child->service_await_join();
        }
    });

    return m_completed_future;
}

void AsyncService::service_await_live()
{
    // std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // // DVLOG(20) << this->debug_prefix() << " entering service_await_live(). State: " << m_state;

    // // We use a CV here to release the lock while we wait for the state to be updated
    // m_cv.wait(lock, [this]() {
    //     // DVLOG(20) << this->debug_prefix() << " checking service_await_live(). State: " << m_state;
    //     return m_state >= AsyncServiceState::Running;
    // });

    // // DVLOG(20) << this->debug_prefix() << " leaving service_await_live(). State: " << m_state;

    // Call get() on the live future to throw any error that occurred during startup
    m_live_promise.get_future().get();
}

void AsyncService::service_stop()
{
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    // Ensures this only gets executed once
    if (this->ensure_state(AsyncServiceState::Stopping))
    {
        // Now, before signaling the stop source, make sure all children are stopped
        for (auto& child : m_children)
        {
            child->service_stop();
        }

        DCHECK(m_stop_source.stop_possible()) << this->debug_prefix() << " Invalid state. Cannot request a stop";

        m_stop_source.request_stop();
    }
}

void AsyncService::service_kill()
{
    // First, make sure we have tried to stop first. Should be a no-op if already stopped
    this->service_stop();

    // Ensures this only gets executed once
    if (this->forward_state(AsyncServiceState::Killing))
    {
        // Kill all children first
        for (auto& child : m_children)
        {
            child->service_kill();
        }

        // Attempt the service specific kill operation
        this->do_service_kill();
    }
}

bool AsyncService::service_await_join(std::chrono::milliseconds wait_duration)
{
    // Guarantee that we set the flag that this was called
    Unwinder ensure_flag([this]() {
        m_service_await_join_called = true;
    });

    {
        std::unique_lock<decltype(m_mutex)> lock(m_mutex);

        // Check to make sure we have started before joining
        CHECK(m_completed_future.valid())
            << this->debug_prefix() << " Must call service_start() before calling service_await_join()";
    }

    // Wait on the completed future (only wait, dont want to throw any exceptions yet)
    if (wait_duration == std::chrono::milliseconds::max())
    {
        m_completed_future.wait();
    }
    else
    {
        if (m_completed_future.wait_for(wait_duration) == boost::fibers::future_status::timeout)
        {
            // Ran out of time. Disarm the unwinder and return false
            ensure_flag.detach();

            return false;
        }
    }

    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    CHECK(m_state >= AsyncServiceState::Completed)
        << this->debug_prefix() << " Invalid state. Was not set to Completed before exiting";

    // Finally, call get() here to rethrow any exceptions in th promise
    m_completed_future.get();

    return true;
}

std::string AsyncService::debug_prefix() const
{
    return MRC_CONCAT_STR("Service[" << m_service_name << "]:");
}

void AsyncService::call_in_destructor()
{
    // Guarantee that we set the flag that this was called
    Unwinder ensure_flag([this]() {
        m_call_in_destructor_called = true;
    });

    auto state = this->state();

    if (state > AsyncServiceState::Initialized)
    {
        if (state == AsyncServiceState::Running)
        {
            LOG(ERROR) << this->debug_prefix()
                       << " service was not stopped/killed before being destructed; issuing kill";
            this->service_kill();
        }

        if (state != AsyncServiceState::Completed)
        {
            LOG(ERROR) << this->debug_prefix() << " service was not joined before being destructed; issuing join";
            service_await_join();
        }
    }

    // Extra careful to lock here just in case
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);

    if (m_completed_future.valid() && !m_service_await_join_called)
    {
        if (m_completed_future.get_exception_ptr())
        {
            LOG(ERROR) << this->debug_prefix()
                       << " An exception was missed. Ensure service_await_join() is called to handle any exceptions";
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
        child->service_await_live();
    }

    // Set the live promise before updating the state so it is ready when the CV is released
    m_live_promise.set_value();

    forward_state(AsyncServiceState::Running, true);
}

SharedFuture<void> AsyncService::completed_future() const
{
    return m_completed_future;
}

void AsyncService::child_service_start_impl(std::shared_ptr<AsyncService> child, bool await_live)
{
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);

        // The state must be running
        CHECK(m_state == AsyncServiceState::Starting || m_state == AsyncServiceState::Running) << "Can only start "
                                                                                                  "child "
                                                                                                  "service in Starting "
                                                                                                  "or "
                                                                                                  "Running state";

        m_children.emplace_back(child);
        child->service_start();
    }

    if (await_live)
    {
        child->service_await_live();
    }
}

void AsyncService::child_service_start_impl(std::unique_ptr<AsyncService>&& child, bool await_live)
{
    CHECK(child) << this->debug_prefix() << " cannot start null child";

    // Convert to a shared pointer so we can save it
    std::shared_ptr<AsyncService> child_shared = std::move(child);

    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);

        const auto& name = child_shared->service_name();

        CHECK(!m_owned_children.contains(name)) << "Child service with name '" << name << "' already added!";

        // Save it to the owned children list
        m_owned_children.emplace(name, child_shared);

        // Now add the child reference
        this->child_service_start_impl(child_shared);
    }

    if (await_live)
    {
        child_shared->service_await_live();
    }
}

bool AsyncService::forward_state(AsyncServiceState new_state, bool assert_forward)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    // State needs to always be moving foward or the same
    CHECK_GE(new_state, m_state) << this->debug_prefix()
                                 << " invalid AsyncServiceState requested; AsyncServiceState is only allowed to "
                                    "advance. Current: "
                                 << m_state << ", Requested: " << new_state;

    if (m_state < new_state)
    {
        DVLOG(20) << this->debug_prefix() << " changing state. From: " << m_state << " to " << new_state;

        m_state = new_state;

        // Notify the CV for anyone waiting on this service
        m_cv.notify_all();

        return true;
    }

    CHECK(!assert_forward) << this->debug_prefix()
                           << " invalid AsyncServiceState requested; AsyncServiceState was required to move forward "
                              "but the state was already set to "
                           << m_state;

    return false;
}

bool AsyncService::ensure_state(AsyncServiceState ensure_state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);

    if (ensure_state > m_state)
    {
        return forward_state(ensure_state);
    }

    return false;
}

void AsyncService::do_service_kill()
{
    // Nothing in base
}

}  // namespace mrc
