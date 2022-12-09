/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

//
// Original Source: https://github.com/boostorg/fiber
//
// Original License:
//          Copyright Oliver Kowalke 2013.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_FIBER_DETAIL_THREAD_BARRIER_H
    #define BOOST_FIBER_DETAIL_THREAD_BARRIER_H

    #include <boost/assert.hpp>
    #include <boost/config.hpp>
    #include <boost/fiber/detail/config.hpp>

    #include <condition_variable>
    #include <cstddef>
    #include <mutex>

    #ifdef BOOST_HAS_ABI_HEADERS
        #include BOOST_ABI_PREFIX
    #endif

// modification from original - namespace
namespace mrc {

class thread_barrier  // NOLINT
{
  private:
    std::size_t initial_;             // NOLINT
    std::size_t current_;             // NOLINT
    bool cycle_{true};                // NOLINT
    std::mutex mtx_{};                // NOLINT
    std::condition_variable cond_{};  // NOLINT

  public:
    explicit thread_barrier(std::size_t initial) : initial_{initial}, current_{initial_}
    {
        BOOST_ASSERT(0 != initial);
    }

    thread_barrier(thread_barrier const&) = delete;
    thread_barrier& operator=(thread_barrier const&) = delete;

    bool wait()
    {
        std::unique_lock<std::mutex> lk(mtx_);
        const bool cycle = cycle_;
        if (0 == --current_)
        {
            cycle_   = !cycle_;
            current_ = initial_;
            lk.unlock();  // no pessimization
            cond_.notify_all();
            return true;
        }
        cond_.wait(lk, [&]() { return cycle != cycle_; });
        return false;
    }
};

}  // namespace mrc

#endif  // BOOST_FIBER_DETAIL_THREAD_BARRIER_H
