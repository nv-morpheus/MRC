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

#include "mrc/runnable/runner.hpp"

#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/engine.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <ext/alloc_traits.h>
#include <glog/logging.h>

#include <atomic>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

// IWYU thinks we need std::max for calling m_instances.resize()
// IWYU pragma: no_include <algorithm>

namespace mrc::runnable {

static std::string runnable_state_str(const Runner::State& state)
{
    switch (state)
    {
    case Runner::State::Unqueued:
        return "Runner::State::Unqueued";
    case Runner::State::Queued:
        return "Runner::State::Queued";
    case Runner::State::Running:
        return "Runner::State::Running";
    case Runner::State::Error:
        return "Runner::State::Error";
    case Runner::State::Completed:
        return "Runner::State::Completed";
    }

    LOG(FATAL) << "unhandled Runner::State value";
    return "Fatal Error: Unhandled Runner::State value";
}

Runner::Runner(std::unique_ptr<Runnable> runnable) : m_runnable(std::move(runnable))
{
    m_runnable->update_state(Runnable::State::Owned);
}

Runner::~Runner()
{
    bool is_running;
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        is_running = (!m_instances.empty());
        m_can_run  = false;  // disable run from being called
    }

    if (is_running)
    {
        m_runnable->update_state(Runnable::State::Kill);
        for (auto& instance : m_instances)
        {
            await_join();
        }
    }
}

void Runner::enqueue(std::shared_ptr<Engines> launcher, std::vector<std::shared_ptr<Context>>&& contexts)
{
    DCHECK(launcher);
    DCHECK_EQ(launcher->size(), contexts.size());
    {
        std::lock_guard<decltype(m_mutex)> lock(m_mutex);
        if (not m_can_run)
        {
            throw exceptions::MrcRuntimeError("Runner::run() is disabled");
        }

        // update to instance count = launcher.count()
        DCHECK_EQ(m_instances.size(), 0);
        m_instances.resize(contexts.size());
        for (int i = 0; i < contexts.size(); ++i)
        {
            m_instances[i].m_uid         = contexts[i]->rank();
            m_instances[i].m_live_future = m_instances[i].m_live_promise.get_future().share();
            m_instances[i].m_context     = contexts[i];
            m_instances[i].m_engine      = launcher->launchers()[i];
            update_state(contexts[i]->rank(), State::Queued);
        }

        // mark runnable as running unless someone has already marked it to as stop or kill
        if (m_runnable->m_state < Runnable::State::Run)
        {
            m_runnable->update_state(Runnable::State::Run);
        }
        else
        {
            LOG(WARNING) << "detected a runnable as being marked stop or kill before it was run";
        }
    }

    CHECK_EQ(m_instances.size(), launcher->launchers().size());

    m_remaining_instances = launcher->size();

    for (auto& instance : m_instances)
    {
        auto context = instance.m_context;
        auto engine  = instance.m_engine;

        auto f = engine->launch_task([this, context, &instance] {
            context->init(*this);
            update_state(context->rank(), State::Running);
            instance.m_live_promise.set_value();
            m_runnable->main(*context);
            if (!context->status())
            {
                update_state(context->rank(), State::Error);
            }
            update_state(context->rank(), State::Completed);
            m_status = m_status && context->status();
            if (--m_remaining_instances == 0)
            {
                if (m_completion_callback)
                {
                    m_completion_callback(m_status);
                }
            }
            context->finish();
        });

        instance.m_join_future = f.share();
    }
}

const std::vector<Runner::Instance>& Runner::instances() const
{
    return m_instances;
}

void Runner::await_live() const
{
    for (const auto& instance : instances())
    {
        instance.live_future().get();
    }
}

void Runner::await_join() const
{
    std::exception_ptr first_exception{nullptr};
    for (const auto& instance : instances())
    {
        try
        {
            instance.join_future().get();
        } catch (...)
        {
            if (first_exception == nullptr)
            {
                first_exception = std::current_exception();
            }
        }
    }
    m_instances.clear();
    if (first_exception)
    {
        LOG(ERROR) << "Runner::await_join - an exception was caught while awaiting on one or more contexts/instances - "
                      "rethrowing";
        std::rethrow_exception(std::move(first_exception));
    }
}

void Runner::stop() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_runnable->update_state(Runnable::State::Stop);
}

void Runner::kill() const
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_runnable->update_state(Runnable::State::Kill);
}

void Runner::update_state(std::size_t launcher_id, State new_state)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    CHECK(m_runnable);
    CHECK_LT(launcher_id, m_instances.size());
    auto& state = m_instances.at(launcher_id).m_state;
    CHECK(state < new_state) << "Runner::State failed to advance in the proper order; current state: "
                             << runnable_state_str(state) << "; target state: " << runnable_state_str(new_state);
    auto old_state = state;
    state          = new_state;
    if (m_on_instance_state_change)
    {
        m_on_instance_state_change(*m_runnable, launcher_id, old_state, new_state);
    }
}

std::size_t Runner::Instance::uid() const
{
    return m_uid;
}

Runner::State Runner::Instance::state() const
{
    return m_state;
}

SharedFuture<void> Runner::Instance::live_future() const
{
    return m_live_future;
}

SharedFuture<void> Runner::Instance::join_future() const
{
    return m_join_future;
}

void Runner::on_instance_state_change_callback(on_instance_state_change_t callback)
{
    CHECK(m_on_instance_state_change == nullptr);
    m_on_instance_state_change = callback;
}

Runnable& Runner::runnable()
{
    CHECK(m_runnable);
    return *m_runnable;
}

void Runner::on_completion_callback(on_completion_callback_t callback)
{
    CHECK(m_completion_callback == nullptr);
    m_completion_callback = callback;
}
}  // namespace mrc::runnable
