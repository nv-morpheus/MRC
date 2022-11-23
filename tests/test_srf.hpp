/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/memory/memory_kind.hpp"
#include "mrc/memory/resources/device/cuda_malloc_resource.hpp"  // IWYU pragma: export
#include "mrc/memory/resources/host/pinned_memory_resource.hpp"  // IWYU pragma: export

#include <glog/logging.h>  // IWYU pragma: keep
#include <gtest/gtest.h>   // IWYU pragma: keep

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>  // for mutex & unique_lock

#define TEST_CLASS(name)                      \
    class Test##name : public ::testing::Test \
    {}

namespace mrc {
// class that records when it's moved/copied
struct CopyMoveCounter
{
  public:
    CopyMoveCounter();

    // Create with value
    CopyMoveCounter(int value);

    CopyMoveCounter(const CopyMoveCounter& o);

    CopyMoveCounter(CopyMoveCounter&& o);

    CopyMoveCounter& operator=(const CopyMoveCounter& o);

    CopyMoveCounter& operator=(CopyMoveCounter&& o);

    std::size_t copy_constructed_count() const;
    std::size_t copy_assignment_count() const;
    std::size_t move_constructed_count() const;
    std::size_t move_assignment_count() const;

    std::size_t copy_count() const;
    std::size_t move_count() const;

    bool was_copied() const;
    bool was_moved() const;

    void inc();

    int value() const;

    static std::atomic<int> global_default_constructed_count;
    static std::atomic<int> global_value_constructed_count;
    static std::atomic<int> global_copy_constructed_count;
    static std::atomic<int> global_move_constructed_count;
    static std::atomic<int> global_copy_assignment_count;
    static std::atomic<int> global_move_assignment_count;

    static void reset();

    static int global_move_count();

    static int global_copy_count();

  private:
    std::size_t m_copy_constructed_count{0};
    std::size_t m_copy_assignment_count{0};
    std::size_t m_move_constructed_count{0};
    std::size_t m_move_assignment_count{0};
    mutable bool m_was_copied{false};
    mutable bool m_was_moved{false};
    int m_value{-1};
};

// class TestCoreResorucesImpl : public core::Resources
// {
//   public:
//     TestCoreResorucesImpl() :
//       m_host_view(std::make_shared<memory::pinned_memory_resource>()),
//       m_device_view(std::make_shared<memory::cuda_malloc_resource>(0))
//     {}
//     ~TestCoreResorucesImpl() override = default;

//     host_view_t host_resource_view() override
//     {
//         return m_host_view;
//     }
//     device_view_t device_resource_view() override
//     {
//         return m_device_view;
//     }

//   private:
//     host_view_t m_host_view;
//     device_view_t m_device_view;
// };

// This class can be used to check whether a certain parallelization is hit. It works similar to a barrier in that it
// blocks threads until a specific number of threads hit the same point. The only difference is this class has a timeout
// option on wait. Only if N parallel threads reach the barrier within a specified timeframe will they be allowed to
// pass. If N+1 or N-1 threads are used, the code will temporarily deadlock until the timeout is reached and false will
// be returned.
class ParallelTester
{
  public:
    ParallelTester(size_t count) : m_count(count) {}

    /**
     * @brief Method to call at the parallelization test point by all threads. Can be used in gtest with
     * `EXPECT_TRUE(parallel_test.wait_for(100ms));` to fail if parallelization isnt met
     *
     * @tparam RepT Duration Rep type
     * @tparam PeriodT Duration Period type
     * @param rel_time A std::chrono::duration object
     * @return true If the parallelization count was met within the timeframe
     * @return false If a deadlock occurred due to an incorrect number of threads
     */
    template <class RepT, class PeriodT>
    bool wait_for(const std::chrono::duration<RepT, PeriodT>& rel_time)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        // Get the current phase
        size_t phase = m_phase;

        if (m_count == ++m_waiting)
        {
            // Reset the waiters
            m_waiting = 0;

            // Increment the phase
            m_phase++;

            // Release those waiting
            m_cv.notify_all();

            return true;
        }

        return m_cv.wait_for(lock, rel_time, [this, &phase] {
            // Block until this phase is over
            return m_phase != phase;
        });
    }

  private:
    size_t m_count;
    size_t m_phase{0};
    size_t m_waiting{0};
    std::mutex m_mutex;
    std::condition_variable m_cv;
};

}  // namespace mrc

using namespace mrc;
