/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <exception>
#include <mutex>
#include <queue>

#pragma once

namespace mrc {

/**
 * @brief A utility for catching out-of-stack exceptions in a thread-safe manner such that they
 * can be checked and throw from a parent thread.
 */
class ExceptionCatcher
{
  public:
    /**
     * @brief "catches" an exception to the catcher
     */
    void push_exception(std::exception_ptr ex);

    /**
     * @brief checks to see if any exceptions have been "caught" by the catcher.
     */
    bool has_exception();

    /**
     * @brief rethrows the next exception (in the order in which it was "caught").
     */
    void rethrow_next_exception();

  private:
    std::mutex m_mutex{};
    std::queue<std::exception_ptr> m_exceptions{};
};

}  // namespace mrc
