/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <mrc/exceptions/exception_catcher.hpp>

namespace mrc {

void ExceptionCatcher::push_exception(std::exception_ptr ex)
{
    auto lock = std::lock_guard(m_mutex);
    m_exceptions.push(ex);
}

bool ExceptionCatcher::has_exception()
{
    auto lock = std::lock_guard(m_mutex);
    return not m_exceptions.empty();
}

void ExceptionCatcher::rethrow_next_exception()
{
    auto lock = std::lock_guard(m_mutex);

    if (m_exceptions.empty())
    {
        return;
    }

    auto ex = m_exceptions.front();

    m_exceptions.pop();

    std::rethrow_exception(ex);
}

}  // namespace mrc
