/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runnable/engine.hpp"

#include "mrc/types.hpp"  // for Future

#include <mutex>    // for mutex, lock_guard
#include <utility>  // for move

namespace mrc::runnable {

Future<void> Engine::launch_task(std::function<void()> task)
{
    std::lock_guard<decltype(m_mutex)> lock(m_mutex);
    m_launched = true;
    return do_launch_task(std::move(task));
}

}  // namespace mrc::runnable
