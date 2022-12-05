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

#include "mrc/coroutines/concepts/awaitable.hpp"
#include "mrc/coroutines/thread_local_context.hpp"

namespace mrc::coroutines {

template <concepts::awaiter T>
struct ResumableAwaiter;

class Resumable
{
  public:
    bool resume()
    {
        if (!m_handle.done())
        {
            if (m_thread_pool != nullptr)
            {
                m_thread_pool->resume(m_handle);
            }
            else
            {
                m_handle.resume();
            }
        }
        return !m_handle.done();
    }

  protected:
    void suspend_coroutine(std::coroutine_handle<> handle);

    void resume_on_thread_pool(ThreadPool* thread_pool);

  private:
    std::coroutine_handle<> m_handle;
    ThreadPool* m_thread_pool;

    template <concepts::awaiter T>
    friend class ResumableAwaiter;
};

template <concepts::awaiter T>
struct ResumableAwaiter : public T
{
    ResumableAwaiter& resume_immediate()
    {
        this->resume_on_thread_pool(nullptr);
        return *this;
    }

    ResumableAwaiter& resume_on(ThreadPool* thread_pool)
    {
        this->resume_on_thread_pool(thread_pool);
        return *this;
    }
};

}  // namespace mrc::coroutines
