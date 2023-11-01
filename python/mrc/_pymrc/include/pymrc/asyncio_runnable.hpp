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

#include "pymrc/asyncio_scheduler.hpp"

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

        auto await_resume() noexcept
        {
            return m_future.get();
        }

      private:
        boost::fibers::future<return_t> m_future;
        std::function<void(std::coroutine_handle<>)> m_inner_fn;
    };

    std::function<SignatureT> m_fn;
};

template <typename T>
class BoostFutureReader
{
  public:
    template <typename FuncT>
    BoostFutureReader(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    coroutines::Task<mrc::channel::Status> async_read(T& value)
    {
        co_return co_await m_awaiter(std::ref(value));
    }

  private:
    BoostFutureAwaitableOperation<mrc::channel::Status(T&)> m_awaiter;
};

template <typename T>
class BoostFutureWriter
{
  public:
    template <typename FuncT>
    BoostFutureWriter(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    coroutines::Task<mrc::channel::Status> async_write(T&& value)
    {
        co_return co_await m_awaiter(std::move(value));
    }

  private:
    BoostFutureAwaitableOperation<mrc::channel::Status(T&&)> m_awaiter;
};

template <typename T>
class CoroutineRunnableSink : public mrc::node::WritableProvider<T>,
                              public mrc::node::ReadableAcceptor<T>,
                              public mrc::node::SinkChannelOwner<T>
{
  protected:
    CoroutineRunnableSink() :
      m_reader([this](T& value) {
          return this->get_readable_edge()->await_read(value);
      })
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    auto build_readable_generator(std::stop_token stop_token) -> mrc::coroutines::AsyncGenerator<T>
    {
        while (!stop_token.stop_requested())
        {
            T value;

            // Pull a message off of the upstream channel
            auto status = co_await m_reader.async_read(std::ref(value));

            if (status != mrc::channel::Status::success)
            {
                break;
            }

            co_yield std::move(value);
        }

        co_return;
    }

  private:
    BoostFutureReader<T> m_reader;
};

template <typename T>
class CoroutineRunnableSource : public mrc::node::WritableAcceptor<T>,
                                public mrc::node::ReadableProvider<T>,
                                public mrc::node::SourceChannelOwner<T>
{
  protected:
    CoroutineRunnableSource()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());

        m_writer = std::make_shared<BoostFutureWriter<T>>([this](T&& value) {
            return this->get_writable_edge()->await_write(std::move(value));
        });
    }

    auto get_writable_receiver() -> std::shared_ptr<BoostFutureWriter<T>>
    {
        return m_writer;
    }

  private:
    std::shared_ptr<BoostFutureWriter<T>> m_writer;
};

template <typename InputT, typename OutputT>
class AsyncioRunnable : public CoroutineRunnableSink<InputT>,
                        public CoroutineRunnableSource<OutputT>,
                        public mrc::runnable::RunnableWithContext<>
{
    using state_t       = mrc::runnable::Runnable::State;
    using task_buffer_t = mrc::coroutines::ClosableRingBuffer<size_t>;

  public:
    AsyncioRunnable(size_t concurrency = 8) : m_concurrency(concurrency){};
    ~AsyncioRunnable() override = default;

  private:
    void run(mrc::runnable::Context& ctx) override;
    void on_state_update(const state_t& state) final;

    coroutines::Task<> main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler);

    coroutines::Task<> process_one(InputT&& value,
                                   std::shared_ptr<BoostFutureWriter<OutputT>> writer,
                                   task_buffer_t& task_buffer,
                                   std::shared_ptr<mrc::coroutines::Scheduler> on,
                                   ExceptionCatcher& catcher);

    virtual mrc::coroutines::AsyncGenerator<OutputT> on_data(InputT&& value) = 0;

    std::stop_source m_stop_source;

    size_t m_concurrency{8};
};

template <typename InputT, typename OutputT>
void AsyncioRunnable<InputT, OutputT>::run(mrc::runnable::Context& ctx)
{
    // auto& scheduler = ctx.scheduler();

    // TODO(MDD): Eventually we should get this from the context object. For now, just create it directly
    auto scheduler = std::make_shared<AsyncioScheduler>(m_concurrency);

    // Now use the scheduler to run the main task until it is complete
    scheduler->run_until_complete(this->main_task(scheduler));

    // Need to drop the output edges
    mrc::node::SourceProperties<OutputT>::release_edge_connection();
    mrc::node::SinkProperties<InputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
coroutines::Task<> AsyncioRunnable<InputT, OutputT>::main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler)
{
    // Get the generator and receiver
    auto input_generator = CoroutineRunnableSink<InputT>::build_readable_generator(m_stop_source.get_token());
    auto output_receiver = CoroutineRunnableSource<OutputT>::get_writable_receiver();

    // Create the task buffer to limit the number of running tasks
    task_buffer_t task_buffer{{.capacity = m_concurrency}};

    size_t i = 0;

    auto iter = co_await input_generator.begin();

    coroutines::TaskContainer outstanding_tasks(scheduler);

    ExceptionCatcher catcher{};

    while (not catcher.has_exception() and iter != input_generator.end())
    {
        // Weird bug, cant directly move the value into the process_one call
        auto data = std::move(*iter);

        // Wait for an available slot in the task buffer
        co_await task_buffer.write(i);

        outstanding_tasks.start(this->process_one(std::move(data), output_receiver, task_buffer, scheduler, catcher));

        // Advance the iterator
        co_await ++iter;
        ++i;
    }

    // Close the buffer
    task_buffer.close();

    // Now block until all tasks are complete
    co_await task_buffer.completed();

    co_await outstanding_tasks.garbage_collect_and_yield_until_empty();

    catcher.rethrow_next_exception();
}

template <typename InputT, typename OutputT>
coroutines::Task<> AsyncioRunnable<InputT, OutputT>::process_one(InputT&& value,
                                                                 std::shared_ptr<BoostFutureWriter<OutputT>> writer,
                                                                 task_buffer_t& task_buffer,
                                                                 std::shared_ptr<mrc::coroutines::Scheduler> on,
                                                                 ExceptionCatcher& catcher)
{
    co_await on->yield();

    try
    {
        // Call the on_data function
        auto on_data_gen = this->on_data(std::move(value));

        auto iter = co_await on_data_gen.begin();

        while (iter != on_data_gen.end())
        {
            // Weird bug, cant directly move the value into the async_write call
            auto data = std::move(*iter);

            co_await writer->async_write(std::move(data));

            // Advance the iterator
            co_await ++iter;
        }
    } catch (...)
    {
        catcher.push_exception(std::current_exception());
    }

    // Return the slot to the task buffer
    co_await task_buffer.read();
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
