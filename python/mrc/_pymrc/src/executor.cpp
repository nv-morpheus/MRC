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

#include "pymrc/executor.hpp"  // IWYU pragma: associated

#include "pymrc/pipeline.hpp"

#include "mrc/pipeline/executor.hpp"
#include "mrc/pipeline/pipeline.hpp"  // IWYU pragma: keep
#include "mrc/pipeline/system.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <chrono>
#include <csignal>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <utility>

// IWYU thinks we need array for calling py::print()
// IWYU pragma: no_include <array>

namespace mrc::pymrc {

namespace py = pybind11;

std::function<void()> create_gil_initializer()
{
    bool has_pydevd_trace = false;

    // We check if there is a debugger by looking at sys.gettrace() and seeing if the function contains 'pydevd'
    // somewhere in the module name. Its important to get this right because calling `debugpy.debug_this_thread()`
    // will fail if there is no debugger and can dramatically alter performanc
    auto sys = pybind11::module_::import("sys");

    auto trace_func = sys.attr("gettrace")();

    if (!trace_func.is_none())
    {
        // Convert it to a string to quickly get its module and name
        auto trace_func_str = pybind11::str(trace_func);

        if (!trace_func_str.attr("find")("pydevd").equal(pybind11::int_(-1)))
        {
            VLOG(10) << "Found pydevd trace function. Will attempt to enable debugging for MRC threads.";
            has_pydevd_trace = true;
        }
    }

    return [has_pydevd_trace] {
        pybind11::gil_scoped_acquire gil;

        // Increment the ref once to prevent creating and destroying the thread state constantly
        gil.inc_ref();

        try
        {
            // Try to load debugpy only if we found a trace function
            if (has_pydevd_trace)
            {
                auto debugpy = pybind11::module_::import("debugpy");

                auto debug_this_thread = debugpy.attr("debug_this_thread");

                debug_this_thread();

                VLOG(10) << "Debugging enabled from mrc threads";
            }
        } catch (pybind11::error_already_set& err)
        {
            if (err.matches(PyExc_ImportError))
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Could import error on debugpy";
                // Fail silently
            }
            else
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Unknown error: " << err.what();
                // Rethrow everything else
                throw;
            }
        }
    };
}

std::function<void()> create_gil_finalizer()
{
    bool python_finalizing = _Py_IsFinalizing() != 0;

    if (python_finalizing)
    {
        // If python if finalizing, dont worry about thread state
        return nullptr;
    }

    // Ensure we dont have the GIL here otherwise this deadlocks.
    return [] {
        bool python_finalizing = _Py_IsFinalizing() != 0;

        if (python_finalizing)
        {
            // If python if finalizing, dont worry about thread state
            return;
        }

        pybind11::gil_scoped_acquire gil;

        // Decrement the ref to destroy the GIL states
        gil.dec_ref();
    };
}

class StopIteration : public py::stop_iteration
{
  public:
    StopIteration(py::object&& result);

    void set_error() const override;  // override

  private:
    py::object m_result;
};

class UniqueLockRelease
{
  public:
    UniqueLockRelease(std::unique_lock<std::mutex>& l) : m_lock(l)
    {
        // Unlock the lock
        m_lock.unlock();
    }

    ~UniqueLockRelease()
    {
        // Relock
        m_lock.lock();
    }

  private:
    std::unique_lock<std::mutex>& m_lock;
};

/** Stop iteration impls -- move to own file **/
StopIteration::StopIteration(py::object&& result) : stop_iteration("--"), m_result(std::move(result)){};
void StopIteration::set_error() const
{
    PyErr_SetObject(PyExc_StopIteration, this->m_result.ptr());
}

/** Awaitable impls -- move to own file **/
Awaitable::Awaitable() = default;

Awaitable::Awaitable(Future<py::object>&& _future) : m_future(std::move(_future)) {}

std::shared_ptr<Awaitable> Awaitable::iter()
{
    return this->shared_from_this();
}

std::shared_ptr<Awaitable> Awaitable::await()
{
    return this->shared_from_this();
}

void Awaitable::next()
{
    // Need to release the GIL before  waiting
    py::gil_scoped_release nogil;

    // check if the future is resolved (with zero timeout)
    auto status = this->m_future.wait_for(std::chrono::milliseconds(0));

    if (status == boost::fibers::future_status::ready)
    {
        // Grab the gil before moving and throwing
        py::gil_scoped_acquire gil;

        // job done -> throw
        auto exception = StopIteration(std::move(this->m_future.get()));

        throw exception;
    }
}

/** Executor impls -- move to own file **/
Executor::Executor(std::shared_ptr<Options> options)
{
    // Before creating the internal exec, set the signal mask so we can capture Ctrl+C
    sigset_t sigset;
    sigset_t pysigset;

    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    auto result = pthread_sigmask(SIG_BLOCK, &sigset, &pysigset);

    // Now create the executor
    auto system = mrc::make_system(options);

    // Now add the gil initializers and finalizers
    system->add_thread_initializer(create_gil_initializer());
    system->add_thread_finalizer(create_gil_finalizer());

    // Must release the GIL while we create the executor
    pybind11::gil_scoped_release nogil;

    m_exec = mrc::make_executor(std::move(system));
}

Executor::~Executor()
{
    // Ensure we have stopped
    // this->stop(); //TODO(MDD): Do we want to call this here? Throws an error if it stopped on its own already

    // before release the executor, we need to drop the gil so that system can acquire the gil in each thread finalizer
    py::gil_scoped_release gil;
    m_exec.reset();

    if (m_join_future.valid() &&
        m_join_future.wait_for(std::chrono::milliseconds(100)) != boost::fibers::future_status::ready)
    {
        py::gil_scoped_acquire gil;

        py::print(
            "Executable was not complete before it was destroyed! Ensure you have called `join()` or `await "
            "join_async()` before the Executor goes out of scope.");
    }
}

void Executor::register_pipeline(pymrc::Pipeline& pipeline)
{
    m_exec->register_pipeline(pipeline.get_wrapped());
}

void Executor::start()
{
    py::gil_scoped_release nogil;

    // Run the start future
    m_exec->start();

    // Now enqueue a join future
    m_join_future = boost::fibers::async([this] {
        m_exec->join();
    });
}

void Executor::stop()
{
    m_exec->stop();
}

void Executor::join()
{
    {
        // Release the GIL before blocking
        py::gil_scoped_release nogil;

        // Wait without the GIL
        m_join_future.wait();
    }

    // Call get() with the GIL to rethrow any exceptions
    m_join_future.get();
}

std::shared_ptr<Awaitable> Executor::join_async()
{
    // No gil here
    Future<py::object> py_fiber_future = boost::fibers::async([this]() -> py::object {
        // Wait for the join future
        this->m_join_future.wait();

        // Grab the GIL to return a py::object
        py::gil_scoped_acquire gil;

        // Once we have the GIL, call get() to propagate any exceptions
        this->m_join_future.get();

        return py::none();
    });

    return std::make_shared<Awaitable>(std::move(py_fiber_future));
}

std::shared_ptr<pipeline::IExecutor> Executor::get_executor() const
{
    return m_exec;
}

/** PyBoostFuture impls -- move to own file **/
PyBoostFuture::PyBoostFuture()
{
    m_future = m_promise.get_future();
}

py::object PyBoostFuture::result()
{
    {
        // Release the GIL until we have a value
        py::gil_scoped_release nogil;

        auto current_thread_id = std::this_thread::get_id();

        m_future.wait();

        auto new_thread_id = std::this_thread::get_id();

        if (current_thread_id != new_thread_id)
        {
            LOG(WARNING) << "Thread IDs different!";
        }
    }

    return m_future.get();
}

py::object PyBoostFuture::py_result()
{
    try
    {
        // Get the result
        return std::move(this->result());
    } catch (py::error_already_set& err)
    {
        LOG(ERROR) << "Exception occurred during Future.result(). Error: " << std::string(err.what());
        throw;
    } catch (std::exception& err)
    {
        LOG(ERROR) << "Exception occurred during Future.result(). Error: " << std::string(err.what());
        return py::none();
    }
}

void PyBoostFuture::set_result(py::object&& obj)
{
    m_promise.set_value(std::move(obj));
}

}  // namespace mrc::pymrc
