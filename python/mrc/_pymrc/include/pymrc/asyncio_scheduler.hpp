/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "pymrc/utilities/object_wrappers.hpp"

#include <boost/fiber/future/async.hpp>
#include <mrc/coroutines/io_scheduler.hpp>
#include <mrc/coroutines/scheduler.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
#include <mrc/coroutines/time.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace mrc::pymrc {

/**
 * @brief A MRC Scheduler which allows resuming C++20 coroutines on an Asyncio event loop.
 */
class AsyncioScheduler : public mrc::coroutines::Scheduler
{
  private:
    class ContinueOnLoopOperation
    {
      public:
        ContinueOnLoopOperation(PyObjectHolder loop) : m_loop(std::move(loop)) {}

        static bool await_ready() noexcept
        {
            return false;
        }

        void await_suspend(std::coroutine_handle<> handle) noexcept
        {
            AsyncioScheduler::resume(m_loop, handle);
        }

        static void await_resume() noexcept {}

      private:
        PyObjectHolder m_loop;
    };

    static void resume(PyObjectHolder loop, std::coroutine_handle<> handle) noexcept
    {
        pybind11::gil_scoped_acquire acquire;
        loop.attr("call_soon_threadsafe")(pybind11::cpp_function([handle]() {
            pybind11::gil_scoped_release release;
            handle.resume();
        }));
    }

  public:
    AsyncioScheduler(PyObjectHolder loop) : m_loop(std::move(loop)) {}

    /**
     * @brief Resumes a coroutine on the scheduler's Asyncio event loop
     */
    void resume(std::coroutine_handle<> handle) noexcept override
    {
        AsyncioScheduler::resume(m_loop, handle);
    }

    /**
     * @brief Suspends the current function and resumes it on the scheduler's Asyncio event loop
     */
    [[nodiscard]] coroutines::Task<> yield() override
    {
        co_await ContinueOnLoopOperation(m_loop);
    }

    [[nodiscard]] coroutines::Task<> yield_for(std::chrono::milliseconds amount) override
    {
        co_await coroutines::IoScheduler::get_instance()->yield_for(amount);
        co_await ContinueOnLoopOperation(m_loop);
    };

    [[nodiscard]] coroutines::Task<> yield_until(mrc::coroutines::time_point_t time) override
    {
        co_await coroutines::IoScheduler::get_instance()->yield_until(time);
        co_await ContinueOnLoopOperation(m_loop);
    };

  private:
    mrc::pymrc::PyHolder m_loop;
};

}  // namespace mrc::pymrc
