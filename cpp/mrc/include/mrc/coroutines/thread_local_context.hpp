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

#pragma once

#include <coroutine>

namespace mrc::coroutines {

class ThreadPool;

/**
 * @brief The SRF Runtime has several third-party dependencies that make use of thread_local storage. Because
 * coroutines can yield execution, other coroutines running on the same thread might modify the thread local storage
 * which would have non-deterministic consequences for the resuming coroutine. Since coroutines can also migrate to
 * other threads, it's important for the awaiter to capture any thread local state so it can be restored regardless of
 * where the coroutine is resumed.
 *
 * This could be used to capture thread local state from libraries like CUDA or OpenTelemetry; however, at present time
 * the cost to capture and restore thread local state far exceeds the benefits.
 *
 * This object will be used as a base for Awaiters in the hope that someday more thread local state can move as a
 * context with a coroutine. In the meantime, ThreadLocalContext does capture one meaning thread local value, i.e. a
 * pointer to the coroutines::ThreadPool that was executing the coroutine upto the point it was suspended. Capturing
 * this information allows us to extend concept::awaiter objects to become concept::resumable objects where the
 * programmer can provide guidance on how and where a coroutine should be resumed.
 */
class ThreadLocalContext
{
  public:
    // use when suspending a coroutine
    void suspend_thread_local_context();

    // use when resuming a coroutine
    void resume_thread_local_context();

  protected:
    // resume a suspended coroutine on either the captured thread_pool or a provided thread_pool
    void resume_coroutine(std::coroutine_handle<> coroutine);

    // set the thread_pool on which to resume the suspended coroutine
    void set_resume_on_thread_pool(ThreadPool* thread_pool);

    // if not nullptr, represents the thread pool on which the caller was executing when the coroutine was suspended
    ThreadPool* thread_pool() const;

  private:
    // Pointer to the active thread pool of the suspended coroutine; null if the coroutines was suspended from thread
    // not in a mrc::coroutines::ThreadPool or if suspend_thread_local_context has not been called
    ThreadPool* m_thread_pool{nullptr};

    // TODO(ryan) - using m_should_resume only in debug mode
    bool m_should_resume{false};

    // We explored the idea of migrating the active cuda device id with coroutine awaiters, but unfortunately,
    // the cost of capturing and resuming the thread_local CUDA device id is far too expensive to int
    // int m_cuda_device_id{0};
};

}  // namespace mrc::coroutines
