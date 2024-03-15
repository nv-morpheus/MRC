#include "mrc/coroutines/scheduler.hpp"

#include <chrono>
#include <coroutine>
#include <queue>

#pragma once

namespace mrc::coroutines {

class TestScheduler : public Scheduler
{
  private:
    struct Operation
    {
      public:
        Operation(TestScheduler* self, std::chrono::time_point<std::chrono::steady_clock> time);

        static constexpr bool await_ready()
        {
            return false;
        }

        void await_suspend(std::coroutine_handle<> handle);

        void await_resume() {}

      private:
        TestScheduler* m_self;
        std::chrono::time_point<std::chrono::steady_clock> m_time;
    };

    using item_t = std::pair<std::coroutine_handle<>, std::chrono::time_point<std::chrono::steady_clock>>;
    struct ItemCompare
    {
        bool operator()(item_t& lhs, item_t& rhs);
    };

    std::priority_queue<item_t, std::vector<item_t>, ItemCompare> m_queue;
    std::chrono::time_point<std::chrono::steady_clock> m_time = std::chrono::steady_clock::now();

  public:

    /**
     * @brief Enqueue's the coroutine handle to be resumed at the current logical time.
     */
    void resume(std::coroutine_handle<> handle) noexcept override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the current logical time.
     */
    mrc::coroutines::Task<> yield() override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the current logica time + the given duration.
     */
    mrc::coroutines::Task<> yield_for(std::chrono::milliseconds time) override;

    /**
     * Suspends the current function and enqueue's it to be resumed at the given logical time.
     */
    mrc::coroutines::Task<> yield_until(std::chrono::time_point<std::chrono::steady_clock> time) override;

    /**
     * Immediately resumes the next-in-queue coroutine handle.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
     */
    bool resume_next();

    /**
     * Immediately resumes next-in-queue coroutines up to the current logical time + the given duration, in-order.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
     */
    bool resume_for(std::chrono::milliseconds time);

    /**
     * Immediately resumes next-in-queue coroutines up to the given logical time.
     *
     *  @return true if more coroutines exist in the queue after resuming, false otherwise.
    */
    bool resume_until(std::chrono::time_point<std::chrono::steady_clock> time);
};

}  // namespace mrc::coroutines
