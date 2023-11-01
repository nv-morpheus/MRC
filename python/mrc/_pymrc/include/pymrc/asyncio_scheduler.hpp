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
    AsyncioScheduler(PyObjectHolder loop) : m_loop(std::move(loop)) {}

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

        // TODO(MDD): Check whether or not we need thread safe version
        m_loop.attr("call_soon_threadsafe")(py::cpp_function([this, handle = std::move(coroutine)]() {
            if (handle.done())
            {
                LOG(WARNING) << "AsyncioScheduler::resume() > Attempted to resume a completed coroutine";
                return;
            }

            py::gil_scoped_release nogil;

            handle.resume();
        }));
    }

  private:
    std::coroutine_handle<> schedule_operation(Operation* operation) override
    {
        this->resume(std::move(operation->m_awaiting_coroutine));

        return std::noop_coroutine();
    }

    mrc::pymrc::PyHolder m_loop;
};

}  // namespace mrc::pymrc
