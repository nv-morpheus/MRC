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

/**
 * Original Source: https://github.com/jbaldwin/libcoro
 * Original License: Apache License, Version 2.0; included below
 */

/**
 * Copyright 2021 Josh Baldwin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mrc/coroutines/thread_local_context.hpp"

#include "mrc/coroutines/thread_pool.hpp"

namespace mrc::coroutines {

void ThreadLocalContext::suspend_thread_local_context()
{
    // suspend the srf context
    m_thread_pool   = ThreadPool::from_current_thread();
    m_should_resume = true;
}

void ThreadLocalContext::resume_thread_local_context()
{
    if (m_should_resume)
    {
        // resume the srf context
        m_should_resume = false;
    }
}

void ThreadLocalContext::resume_coroutine(std::coroutine_handle<> coroutine)
{
    if (m_thread_pool != nullptr)
    {
        // add event - scheduled on
        m_thread_pool->resume(coroutine);
        return;
    }

    // add a span since the current execution context will be suspended and the coroutine will be resumed
    ThreadLocalContext ctx;
    ctx.suspend_thread_local_context();
    coroutine.resume();
    ctx.resume_thread_local_context();
}

void ThreadLocalContext::set_resume_on_thread_pool(ThreadPool* thread_pool)
{
    m_thread_pool = thread_pool;
}

ThreadPool* ThreadLocalContext::thread_pool() const
{
    return m_thread_pool;
}

}  // namespace mrc::coroutines
