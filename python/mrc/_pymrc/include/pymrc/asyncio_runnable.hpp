#pragma once

#include "pymrc/asyncio_scheduler.hpp"

#include <boost/fiber/future/async.hpp>
#include <mrc/channel/buffered_channel.hpp>
#include <mrc/channel/status.hpp>
#include <mrc/coroutines/async_generator.hpp>
#include <mrc/coroutines/closable_ring_buffer.hpp>
#include <mrc/coroutines/task.hpp>
#include <mrc/coroutines/task_container.hpp>
#include <mrc/node/sink_properties.hpp>
#include <mrc/runnable/forward.hpp>

#include <coroutine>
#include <exception>
#include <functional>

namespace mrc::pymrc {

template <typename T>
using Task = mrc::coroutines::Task<T>;

class ExceptionCatcher
{
  public:
    void set_exception(std::exception_ptr ex)
    {
        auto lock = std::lock_guard(m_mutex);
        m_exceptions.push(ex);
    }

    bool has_exception()
    {
        auto lock = std::lock_guard(m_mutex);
        return not m_exceptions.empty();
    }

    void rethrow_next_exception()
    {
        auto lock = std::lock_guard(m_mutex);

        if (m_exceptions.empty())
        {
            return;
        }

        auto ex = m_exceptions.front();
        m_exceptions.pop();

        std::rethrow_exception(ex);
    }

  private:
    std::mutex m_mutex{};
    std::queue<std::exception_ptr> m_exceptions{};
};

template <typename SignatureT>
class BoostFutureAwaiter
{
    class Awaiter;

  public:
    BoostFutureAwaiter(std::function<SignatureT> fn) : m_fn(std::move(fn)) {}

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

        bool await_suspend(std::coroutine_handle<> continuation) noexcept
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

            return true;
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
class IReadable
{
  public:
    virtual ~IReadable()                                    = default;
    virtual Task<mrc::channel::Status> async_read(T& value) = 0;
};

template <typename T>
class BoostFutureReader : public IReadable<T>
{
  public:
    template <typename FuncT>
    BoostFutureReader(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    Task<mrc::channel::Status> async_read(T& value) override
    {
        co_return co_await m_awaiter(std::ref(value));
    }

  private:
    BoostFutureAwaiter<mrc::channel::Status(T&)> m_awaiter;
};

template <typename T>
class IWritable
{
  public:
    virtual ~IWritable()                                      = default;
    virtual Task<mrc::channel::Status> async_write(T&& value) = 0;
};

template <typename T>
class BoostFutureWriter : public IWritable<T>
{
  public:
    template <typename FuncT>
    BoostFutureWriter(FuncT&& fn) : m_awaiter(std::forward<FuncT>(fn))
    {}

    Task<mrc::channel::Status> async_write(T&& value) override
    {
        co_return co_await m_awaiter(std::move(value));
    }

  private:
    BoostFutureAwaiter<mrc::channel::Status(T&&)> m_awaiter;
};

template <typename T>
class CoroutineRunnableSink : public mrc::node::WritableProvider<T>,
                              public mrc::node::ReadableAcceptor<T>,
                              public mrc::node::SinkChannelOwner<T>
{
  protected:
    CoroutineRunnableSink()
    {
        // Set the default channel
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    auto build_readable_generator(std::stop_token stop_token) -> mrc::coroutines::AsyncGenerator<T>
    {
        auto read_awaiter = BoostFutureReader<T>([this](T& value) {
            return this->get_readable_edge()->await_read(value);
        });

        while (!stop_token.stop_requested())
        {
            T value;

            // Pull a message off of the upstream channel
            auto status = co_await read_awaiter.async_read(std::ref(value));

            if (status != mrc::channel::Status::success)
            {
                break;
            }

            co_yield std::move(value);
        }

        co_return;
    }
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
    }

    // auto build_readable_generator(std::stop_token stop_token)
    //     -> mrc::coroutines::AsyncGenerator<mrc::coroutines::detail::VoidValue>
    // {
    //     while (!stop_token.stop_requested())
    //     {
    //         co_yield mrc::coroutines::detail::VoidValue{};
    //     }

    //     co_return;
    // }

    auto build_writable_receiver() -> std::shared_ptr<IWritable<T>>
    {
        return std::make_shared<BoostFutureWriter<T>>([this](T&& value) {
            return this->get_writable_edge()->await_write(std::move(value));
        });
    }
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

    Task<void> main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler);

    Task<void> process_one(InputT&& value,
                           std::shared_ptr<IWritable<OutputT>> writer,
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
    mrc::node::SourceProperties<InputT>::release_edge_connection();
    mrc::node::SinkProperties<OutputT>::release_edge_connection();
}

template <typename InputT, typename OutputT>
Task<void> AsyncioRunnable<InputT, OutputT>::main_task(std::shared_ptr<mrc::coroutines::Scheduler> scheduler)
{
    // Get the generator and receiver
    auto input_generator = CoroutineRunnableSink<InputT>::build_readable_generator(m_stop_source.get_token());
    auto output_receiver = CoroutineRunnableSource<OutputT>::build_writable_receiver();

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
Task<void> AsyncioRunnable<InputT, OutputT>::process_one(InputT&& value,
                                                         std::shared_ptr<IWritable<OutputT>> writer,
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
        // TODO(cwharris): communicate error back to the runnable's main main task
        catcher.set_exception(std::current_exception());
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
