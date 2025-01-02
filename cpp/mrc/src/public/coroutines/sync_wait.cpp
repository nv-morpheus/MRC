/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/coroutines/sync_wait.hpp"

namespace mrc::coroutines::detail {

SyncWaitEvent::SyncWaitEvent(bool initially_set) : m_set(initially_set) {}

auto SyncWaitEvent::set() noexcept -> void
{
    {
        std::lock_guard<std::mutex> g{m_mutex};
        m_set = true;
    }

    m_cv.notify_all();
}

auto SyncWaitEvent::reset() noexcept -> void
{
    std::lock_guard<std::mutex> g{m_mutex};
    m_set = false;
}

auto SyncWaitEvent::wait() noexcept -> void
{
    std::unique_lock<std::mutex> lk{m_mutex};
    m_cv.wait(lk, [this] {
        return m_set;
    });
}

}  // namespace mrc::coroutines::detail
