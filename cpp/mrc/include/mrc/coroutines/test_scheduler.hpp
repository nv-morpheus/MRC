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
    void resume(std::coroutine_handle<> handle) noexcept override;

    mrc::coroutines::Task<> yield() override;

    mrc::coroutines::Task<> yield_for(std::chrono::milliseconds time) override;

    mrc::coroutines::Task<> yield_until(std::chrono::time_point<std::chrono::steady_clock> time) override;

    bool resume_next();

    bool resume_for(std::chrono::milliseconds time);

    bool resume_until(std::chrono::time_point<std::chrono::steady_clock> time);
};

}  // namespace mrc::coroutines
