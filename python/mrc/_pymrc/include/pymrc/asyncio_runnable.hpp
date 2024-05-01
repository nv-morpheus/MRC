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

#include "pymrc/asyncio_scheduler.hpp"
#include "pymrc/edge_adapter.hpp"
#include "pymrc/node.hpp"
#include "pymrc/utilities/object_wrappers.hpp"

#include <boost/fiber/future/async.hpp>
#include <mrc/channel/buffered_channel.hpp>
#include <mrc/channel/status.hpp>
#include <mrc/coroutines/async_generator.hpp>
#include <mrc/coroutines/closable_ring_buffer.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
#include <mrc/exceptions/exception_catcher.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/runnable/forward.hpp>

#include <coroutine>
#include <exception>
#include <functional>

namespace mrc::pymrc {

/**
 * @brief A wrapper for executing a function as an async boost fiber, the result of which is a
 * C++20 coroutine awaiter.
 */
template <typename SignatureT>
class BoostFutureAwaitableOperation
{
    class Awaiter;

  public:
    BoostFutureAwaitableOperation(std::function<SignatureT> fn) : m_fn(std::move(fn)) {}

    /**
     * @brief Calls the wrapped function as an asyncboost fiber and returns a C++20 coroutine awaiter.
     */
    template <typename... ArgsT>
    auto operator()(ArgsT&&... args) -> Awaiter
    {
        // Make a copy of m_fn here so we can call this operator again
        return Awaiter(m_fn, std::forward<ArgsT>(args)...);
    }

  private:
    class Awaiter
    {
      public:
        using return_t = typename std::function<SignatureT>::result_type;

        template <typename... ArgsT>
        Awaiter(std::function<SignatureT> fn, ArgsT&&... args)
        {
            m_future = boost::fibers::async(boost::fibers::launch::post, fn, std::forward<ArgsT>(args)...);
        }

        bool await_ready() noexcept
        {
            return false;
        }

        void await_suspend(std::coroutine_handle<> continuation) noexcept
        {
            // Launch a new fiber that waits on the future and then resumes the coroutine
            boost::fibers::async(
                boost::fibers::launch::post,
                [this](std::coroutine_handle<> continuation) {
                    // Wait on the future
                    m_future.wait();

                    // Resume the coroutine
                    continuation.resume();
                },
                std::move(continuation));
        }

        auto await_resume()
        {
            return m_future.get();
        }

      private:
        boost::fibers::future<return_t> m_future;
        std::function<void(std::coroutine_handle<>)> m_inner_fn;
    };

    std::function<SignatureT> m_fn;
};

/**
 * @brief A MRC Sink which receives from a channel using an awaitable interface.
 */
template <typename T>
class AsyncSink : public mrc::node::WritableProvider<T>,
                  public mrc::node::ReadableAcceptor<T>,
                  public mrc::node::SinkChannelOwner<T>,
                  public pymrc::AutoRegSinkAdapter<T>,
                  public pymrc::AutoRegEgressPort<T>
{
  protected:
    AsyncSink() :
      m_read_async([this](T& value) {
          return this->get_readable_edge()->await_read(value);
      })
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    /**
     * @brief Asynchronously reads a value from the sink's channel
     */
    coroutines::Task<mrc::channel::Status> read_async(T& value)
    {
        co_return co_await m_read_async(std::ref(value));
    }

  private:
    BoostFutureAwaitableOperation<mrc::channel::Status(T&)> m_read_async;
};

/**
 * @brief A MRC Source which produces to a channel using an awaitable interface.
 */
template <typename T>
class AsyncSource : public mrc::node::WritableAcceptor<T>,
                    public mrc::node::ReadableProvider<T>,
                    public mrc::node::SourceChannelOwner<T>,
                    public pymrc::AutoRegSourceAdapter<T>,
                    public pymrc::AutoRegIngressPort<T>
{
  protected:
    AsyncSource() :
      m_write_async([this](T&& value) {
          return this->get_writable_edge()->await_write(std::move(value));
      })
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    /**
     * @brief Asynchronously writes a value to the source's channel
     */
    coroutines::Task<mrc::channel::Status> write_async(T&& value)
    {
        co_return co_await m_write_async(std::move(value));
    }

  private:
    BoostFutureAwaitableOperation<mrc::channel::Status(T&&)> m_write_async;
};

/**
 * @brief A MRC Runnable base class which hosts it's own asyncio loop and exposes a flatmap hook
 */
template <typename InputT, typename OutputT>
class AsyncioRunnable : public AsyncSink<InputT>,
                        public AsyncSource<OutputT>,
                        public mrc::runnable::RunnableWithContext<>
{
    using state_t       = mrc::runnable::Runnable::State;
    using task_buffer_t = mrc::coroutines::ClosableRingBuffer<size_t>;

  public:
    ~AsyncioRunnable() override = default;

  private:
    /**
     * @brief Runnable's entrypoint.
     */
    void run(mrc::runnable::Context& ctx) override;

    /**
     * @brief Runnable's state control, for stopping from MRC.
     */
    void on_state_update(const state_t& state) final;

    /**
     * @brief The top-level coroutine which is run while the asyncio event loop is running.
     */
    coroutines::Task<> main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler);

    /**
     * @brief The per-value coroutine run asynchronously alongside other calls.
     */
    coroutines::Task<> process_one(InputT value,
                                   std::shared_ptr<mrc::coroutines::Scheduler> on,
                                   ExceptionCatcher& catcher);

    /**
     * @brief Value's read from the sink's channel are fed to this function and yields from the
     * resulting generator are written to the source's channel.
     */
    virtual mrc::coroutines::AsyncGenerator<OutputT> on_data(InputT&& value,
                                                             std::shared_ptr<mrc::coroutines::Scheduler> on) = 0;

    std::stop_source m_stop_source;
};

template <typename InputT, typename OutputT>
void AsyncioRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    std::exception_ptr exception;

    {
        py::gil_scoped_acquire gil;

        auto asyncio = py::module_::import("asyncio");

        auto loop = [](auto& asyncio) -> PyObjectHolder {
            try
            {
                return asyncio.attr("get_running_loop")();
            } catch (...)
            {
                return py::none();
            }
        }(asyncio);

        if (not loop.is_none())
        {
            throw std::runtime_error("asyncio loop already running, but runnable is expected to create it.");
        }

        // Need to create a loop
        DVLOG(10) << "AsyncioRunnable::run() > Creating new event loop";

        // Gets (or more likely, creates) an event loop and runs it forever until stop is called
        loop = asyncio.attr("new_event_loop")();

        // Set the event loop as the current event loop
        asyncio.attr("set_event_loop")(loop);

        // TODO(MDD): Eventually we should get this from the context object. For now, just create it directly
        auto scheduler = std::make_shared<AsyncioScheduler>(loop);

        auto py_awaitable = coro::BoostFibersMainPyAwaitable(this->main_task(scheduler));

        DVLOG(10) << "AsyncioRunnable::run() > Calling run_until_complete() on main_task()";

        try
        {
            loop.attr("run_until_complete")(std::move(py_awaitable));
        } catch (...)
        {
            exception = std::current_exception();
        }

        loop.attr("close")();
    }

    // Sync all progress engines if there are more than one
    ctx.barrier();

    // Only drop the output edges if we are rank 0
    if (ctx.rank() == 0)
    {
        // Need to drop the output edges
        mrc::node::SourceProperties<OutputT>::release_edge_connection();
        mrc::node::SinkProperties<InputT>::release_edge_connection();
    }

    if (exception != nullptr)
    {
        std::rethrow_exception(exception);
    }
}

template <typename InputT, typename OutputT>
coroutines::Task<> AsyncioRunnable<InputT, OutputT>::main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler)
{
    coroutines::TaskContainer outstanding_tasks(scheduler, 8);

    ExceptionCatcher catcher{};

    while (not m_stop_source.stop_requested() and not catcher.has_exception())
    {
        InputT data;

        auto read_status = co_await this->read_async(data);

        if (read_status != mrc::channel::Status::success)
        {
            break;
        }

        outstanding_tasks.start(this->process_one(std::move(data), scheduler, catcher));
    }

    co_await outstanding_tasks.garbage_collect_and_yield_until_empty();

    catcher.rethrow_next_exception();
}

template <typename InputT, typename OutputT>
coroutines::Task<> AsyncioRunnable<InputT, OutputT>::process_one(InputT value,
                                                                 std::shared_ptr<mrc::coroutines::Scheduler> on,
                                                                 ExceptionCatcher& catcher)
{
    co_await on->yield();

    try
    {
        // Call the on_data function
        auto on_data_gen = this->on_data(std::move(value), on);

        auto iter = co_await on_data_gen.begin();

        while (iter != on_data_gen.end())
        {
            // Weird bug, cant directly move the value into the async_write call
            auto data = std::move(*iter);

            co_await this->write_async(std::move(data));

            // Advance the iterator
            co_await ++iter;
        }
    } catch (...)
    {
        catcher.push_exception(std::current_exception());
    }
}

template <typename InputT, typename OutputT>
void AsyncioRunnable<InputT, OutputT>::on_state_update(const state_t& state)
{
    switch (state)
    {
    case state_t::Stop:
        // Do nothing, we wait for the upstream channel to return closed
        // m_stop_source.request_stop();
        break;

    case state_t::Kill:
        m_stop_source.request_stop();
        break;

    default:
        break;
    }
}

}  // namespace mrc::pymrc
