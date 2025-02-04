/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/ucx/common.hpp"

namespace mrc::ucx {

template <typename T>
class Primitive : public std::enable_shared_from_this<Primitive<T>>
{
  public:
    virtual ~Primitive() = default;

    T& handle()
    {
        return m_handle;
    }
    const T& handle() const
    {
        return m_handle;
    }

  protected:
    virtual void init() {}
    virtual void finalize() {}

    T m_handle;
};

}  // namespace mrc::ucx
