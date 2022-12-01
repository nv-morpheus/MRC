/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>

namespace mrc::core {

class SpinLockMutex
{
  public:
    // todo(clang-format-15)
    // clang-format off
    SpinLockMutex() noexcept                                = default;
    ~SpinLockMutex() noexcept                               = default;
    SpinLockMutex(const SpinLockMutex&)                     = delete;
    SpinLockMutex& operator=(const SpinLockMutex&)          = delete;
    SpinLockMutex& operator=(const SpinLockMutex&) volatile = delete;
    // clang-format off

    static inline void yield() noexcept
    {
#if defined(_MSC_VER)
        YieldProcessor();
#elif defined(__i386__) || defined(__x86_64__)
    #if defined(__clang__)
        _mm_pause();  // NOLINT
    #else
        __builtin_ia32_pause();  // NOLINT
    #endif
#elif defined(__arm__)
        __asm__ volatile("yield" ::: "memory");
#else
        static_assert(false, "yield not implemented on unknown architecture");
#endif
    }

    inline bool try_lock() noexcept
    {
        return !m_flag.load(std::memory_order_relaxed) && !m_flag.exchange(true, std::memory_order_acquire);
    }

    void lock() noexcept
    {
        while (true)
        {
            if (!m_flag.exchange(true, std::memory_order_acquire))
            {
                return;
            }
            while (true)
            {
                if (try_lock())
                {
                    return;
                }
                yield();
            }
        }
    }

    void unlock() noexcept
    {
        m_flag.store(false, std::memory_order_release);
    }

  private:
    std::atomic<bool> m_flag{false};
};

}  // namespace mrc::core
