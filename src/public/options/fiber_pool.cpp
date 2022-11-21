/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "srf/options/fiber_pool.hpp"

namespace srf {

FiberPoolOptions& FiberPoolOptions::enable_memory_binding(bool default_true)
{
    m_enable_memory_binding = default_true;
    return *this;
}
FiberPoolOptions& FiberPoolOptions::enable_thread_binding(bool default_true)
{
    m_enable_thread_binding = default_true;
    return *this;
}
FiberPoolOptions& FiberPoolOptions::enable_tracing_scheduler(bool default_false)
{
    m_enable_tracing_scheduler = false;
    return *this;
}
bool FiberPoolOptions::enable_memory_binding() const
{
    return m_enable_memory_binding;
}
bool FiberPoolOptions::enable_thread_binding() const
{
    return m_enable_thread_binding;
}
bool FiberPoolOptions::enable_tracing_scheduler() const
{
    return m_enable_tracing_scheduler;
}

}  // namespace srf
