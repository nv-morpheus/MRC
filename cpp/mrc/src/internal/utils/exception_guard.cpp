/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/utils/exception_guard.hpp"

#include <exception>

namespace mrc::internal::utils {

ExceptionGuard::ExceptionGuard(std::function<void()> lambda)
{
    try
    {
        lambda();
    } catch (...)
    {
        // LOG(INFO) << "exception caught by exception guard - will rethrow on destructor";
        m_ptr = std::current_exception();
    }
}

ExceptionGuard::~ExceptionGuard()
{
    if (m_ptr)
    {
        std::rethrow_exception(m_ptr);
    }
}

}  // namespace mrc::internal::utils
