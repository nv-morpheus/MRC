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

#include <boost/fiber/future/async.hpp>
#include <pysrf/executor.hpp>

#include <pysrf/pipeline.hpp>
#include <pysrf/system.hpp>

#include <srf/core/executor.hpp>
#include <srf/core/utils.hpp>  // for SRF_UNWIND_AUTO, Unwinder
#include <srf/options/options.hpp>
#include <srf/types.hpp>  // for Future, SharedFuture

#include <boost/fiber/future/future.hpp>         // for task_base<>::ptr_type, future
#include <boost/fiber/future/future_status.hpp>  // for future_status, future_status::ready

#include <glog/logging.h>

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <sys/prctl.h>

#include <atomic>
#include <chrono>     // for milliseconds
#include <csignal>    // for siginfo_t
#include <ctime>      // for timespec
#include <exception>  // for exception, exception_ptr
#include <future>
#include <memory>
#include <mutex>
#include <ostream>  // for glog macros
#include <string>
#include <thread>  // for get_id, operator!=, thread::id
#include <utility>

// IWYU thinks we need array for calling py::print()
// IWYU pragma: no_include <array>

// IWYU thinks we need system_error for ft_signal_handler
// IWYU pragma: no_include <system_error>

// IWYU thinks we need map & max for gil_futures
// IWYU pragma: no_include <algorithm>
// IWYU pragma: no_include <map>

namespace srf::pysrf {

namespace py = pybind11;

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
    auto system    = std::make_unique<System>(options);
    auto resources = std::make_unique<SystemResources>(std::move(system));
    m_exec         = std::make_shared<srf::Executor>(std::move(resources));
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

void Executor::register_pipeline(pysrf::Pipeline& pipeline)
{
    m_exec->register_pipeline(pipeline.swap());
}

void Executor::start()
{
    py::gil_scoped_release nogil;

    // Run the start future
    m_exec->start();

    // Now enqueue a join future
    m_join_future = boost::fibers::async([this] { m_exec->join(); });
}

void Executor::stop()
{
    m_exec->stop();
}

void Executor::join()
{
    // this might be all we need here
    // py::gil_scoped_release nogil;
    // m_join_future.get();
    // below this could go

    // Ensure we have the GIL
    py::gil_scoped_acquire gil;

    // block signals in this thread and subsequently
    // spawned threads
    sigset_t sigset;
    sigset_t pysigset;
    sigemptyset(&sigset);
    sigaddset(&sigset, SIGINT);
    sigaddset(&sigset, SIGTERM);
    auto result = pthread_sigmask(SIG_BLOCK, &sigset, &pysigset);

    // Before we exit, we must reset the signal state
    SRF_UNWIND_AUTO(([&pysigset]() {
        // Restore the mask
        pthread_sigmask(SIG_SETMASK, &pysigset, nullptr);
    }));

    std::atomic<bool> executor_running(true);
    std::mutex cv_mutex;

    auto signal_handler = [&executor_running, &cv_mutex, &sigset, this]() {
        prctl(PR_SET_NAME, "SignalHandler", 0, 0, 0);

        // lock prevents modifying value while in body of loop
        std::unique_lock lock(cv_mutex);

        int signal_value = -1;
        int signal_count = 0;

        siginfo_t info;
        // Timeout of 100ms
        struct timespec ts = {0, 100000};

        while (executor_running.load())
        {
            {
                // Release the lock while we wait for the signal
                UniqueLockRelease nolock(lock);

                // Get the signal value and immediately reacquire the lock
                signal_value = sigtimedwait(&sigset, &info, &ts);
            }

            // Check if we received the signal
            if (signal_value >= 0)
            {
                signal_count++;

                // Get the GIL here since we always will print something
                py::gil_scoped_acquire gil;

                if (signal_count == 1)
                {
                    // First signal, stop the executor to allow nice shutdown
                    py::print("Stopping Executor. Waiting for safe shutdown... Press Ctrl+C again to exit");

                    this->stop();
                }
                else
                {
                    // Second time its been hit
                    py::print("Stopping Executor. Waiting for safe shutdown... Press Ctrl+C again to exit");

                    // TODO(MDD): Call kill()

                    // Important part is to stop both loops here
                    executor_running.store(false);
                }
            }
        }
    };

    auto ft_signal_handler = std::async(std::launch::async, signal_handler);

    boost::fibers::future_status status;

    // This is to hold any error that gets caught from a signal
    std::exception_ptr exc_ptr;

    py::gil_scoped_release nogil;

    {
        // lock prevents modifying value while in body of loop
        std::unique_lock lock(cv_mutex);

        // Only way to exit this loop is by setting executor_running = false
        while (executor_running.load())
        {
            {
                // Release the lock while we wait for the signal
                UniqueLockRelease nolock(lock);

                // Get the result and immedately reacquire the lock
                status = this->m_join_future.wait_for(std::chrono::milliseconds(100));
            }

            if (status == boost::fibers::future_status::ready)
            {
                // Set the running flag to false
                executor_running.store(false);
            }
        }
    }

    // Now wait on the started thread. Must not hold lock!
    ft_signal_handler.wait();

    // do
    // {
    //     // Check for a signal
    //     if (PyErr_CheckSignals() != 0)
    //     {
    //         if (!exc_ptr)
    //         {
    //             py::print("Stopping Executor. Waiting for safe shutdown... Press Ctrl+C again to exit");

    //             exc_ptr = std::make_exception_ptr(py::error_already_set());

    //             // Stop
    //             this->stop();
    //         }
    //         else
    //         {
    //             // Second time its been hit
    //             py::print("Stopping Executor. Waiting for safe shutdown... Press Ctrl+C again to exit");

    //             // TODO(MDD): Call kill()
    //             break;
    //         }
    //     }

    //     py::gil_scoped_release nogil;

    //     status = this->m_join_future.wait_for(std::chrono::milliseconds(100));
    // } while (status != boost::fibers::future_status::ready);

    // if (exc_ptr)
    // {
    //     std::rethrow_exception(exc_ptr);
    // }
}

std::shared_ptr<Awaitable> Executor::join_async()
{
    // No gil here

    Future<py::object> py_fiber_future = boost::fibers::async([this]() -> py::object {
        // Wait for the join future
        this->m_join_future.wait();

        // Grab the GIL to return a py::object
        py::gil_scoped_acquire gil;

        return py::none();
    });

    return std::make_shared<Awaitable>(std::move(py_fiber_future));
}

std::shared_ptr<srf::Executor> Executor::get_executor() const
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

}  // namespace srf::pysrf
