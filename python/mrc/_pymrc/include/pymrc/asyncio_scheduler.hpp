/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/coro.hpp"
#include "pymrc/utilities/acquire_gil.hpp"

#include <boost/fiber/future/async.hpp>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <stdexcept>

namespace py = pybind11;

namespace mrc::pymrc {
class AsyncioScheduler : public mrc::coroutines::Scheduler
{
  public:
    std::string description() const override
    {
        return "AsyncioScheduler";
    }

    void resume(std::coroutine_handle<> coroutine) override
    {
        if (coroutine.done())
        {
            LOG(WARNING) << "AsyncioScheduler::resume() > Attempted to resume a completed coroutine";
            return;
        }

        py::gil_scoped_acquire gil;

        auto& loop = this->get_loop();

        // TODO(MDD): Check whether or not we need thread safe version
        loop.attr("call_soon_threadsafe")(py::cpp_function([this, handle = std::move(coroutine)]() {
            if (handle.done())
            {
                LOG(WARNING) << "AsyncioScheduler::resume() > Attempted to resume a completed coroutine";
                return;
            }

            py::gil_scoped_release nogil;

            handle.resume();
        }));
    }

    mrc::pymrc::PyHolder& init_loop()
    {
        CHECK_EQ(PyGILState_Check(), 1) << "Must have the GIL when calling AsyncioScheduler::init_loop()";

        std::unique_lock lock(m_mutex);

        if (m_loop)
        {
            return m_loop;
        }

        auto asyncio_mod = py::module_::import("asyncio");

        auto loop = [asyncio_mod]() -> py::object {
            try
            {
                return asyncio_mod.attr("get_running_loop")();
            } catch (...)
            {
                return py::none();
            }
        }();

        if (not loop.is_none())
        {
            throw std::runtime_error("asyncio loop already running, but runnable is expected to create it.");
        }

        // Need to create a loop
        LOG(INFO) << "AsyncioScheduler::run() > Creating new event loop";

        // Gets (or more likely, creates) an event loop and runs it forever until stop is called
        m_loop = asyncio_mod.attr("new_event_loop")();

        // Set the event loop as the current event loop
        asyncio_mod.attr("set_event_loop")(m_loop);

        return m_loop;
    }

    // Runs the task until its complete
    void run_until_complete(coroutines::Task<>&& task)
    {
        mrc::pymrc::AcquireGIL gil;

        auto& loop = this->init_loop();

        LOG(INFO) << "AsyncioScheduler::run() > Calling run_until_complete() on main_task()";

        // Use the BoostFibersMainPyAwaitable to allow fibers to be progressed
        loop.attr("run_until_complete")(mrc::pymrc::coro::BoostFibersMainPyAwaitable(std::move(task)));
    }

  private:
    std::coroutine_handle<> schedule_operation(Operation* operation) override
    {
        this->resume(std::move(operation->m_awaiting_coroutine));

        return std::noop_coroutine();
    }

    mrc::pymrc::PyHolder& get_loop()
    {
        if (!m_loop)
        {
            throw std::runtime_error("Must call init_loop() before get_loop()");
        }

        // TODO(MDD): Check that we are on the same thread as the loop
        return m_loop;
    }

    std::mutex m_mutex;

    std::atomic_size_t m_outstanding{0};

    mrc::pymrc::PyHolder m_loop;
};

}  // namespace mrc::pymrc
