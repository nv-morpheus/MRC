#include "mrc/coroutines/test_scheduler.hpp"

namespace mrc::coroutines {

TestScheduler::Operation::Operation(TestScheduler* self, std::chrono::time_point<std::chrono::steady_clock> time) :
  m_self(self),
  m_time(time)
{}

bool TestScheduler::ItemCompare::operator()(item_t& lhs, item_t& rhs)
{
    return lhs.second > rhs.second;
}

void TestScheduler::Operation::await_suspend(std::coroutine_handle<> handle)
{
    m_self->m_queue.emplace(std::move(handle), m_time);
}

void TestScheduler::resume(std::coroutine_handle<> handle) noexcept
{
    m_queue.emplace(std::move(handle), std::chrono::steady_clock::now());
}

mrc::coroutines::Task<> TestScheduler::yield()
{
    co_return co_await TestScheduler::Operation{this, m_time};
}

mrc::coroutines::Task<> TestScheduler::yield_for(std::chrono::milliseconds time)
{
    co_return co_await TestScheduler::Operation{this, m_time + time};
}

mrc::coroutines::Task<> TestScheduler::yield_until(std::chrono::time_point<std::chrono::steady_clock> time)
{
    co_return co_await TestScheduler::Operation{this, time};
}

bool TestScheduler::resume_next()
{
    if (m_queue.empty())
    {
        return false;
    }

    auto handle = m_queue.top();

    m_queue.pop();

    m_time = handle.second;

    handle.first.resume();

    return true;
}

bool TestScheduler::resume_for(std::chrono::milliseconds time)
{
    return resume_until(m_time + time);
}

bool TestScheduler::resume_until(std::chrono::time_point<std::chrono::steady_clock> time)
{
    m_time = time;

    while (not m_queue.empty())
    {
        if (m_queue.top().second <= m_time)
        {
            m_queue.top().first.resume();
            m_queue.pop();
        }
        else
        {
            return true;
        }
    }

    return false;
}

}  // namespace mrc::coroutines
