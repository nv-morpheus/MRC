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

#pragma once

#include "mrc/node/operators/operator.hpp"

namespace mrc::node {

template <typename T>
class OperatorComponent final : public Operator<T>
{
  public:
    OperatorComponent(
        std::function<mrc::channel::Status(T&&)> on_next, std::function<void()> on_complete = [] {}) :
      m_on_next(std::move(on_next)),
      m_on_complete(std::move(on_complete))
    {}

  private:
    mrc::channel::Status on_next(T&& obj) final
    {
        return m_on_next(std::move(obj));
    }

    void on_complete() final
    {
        m_on_complete();
    }

    std::function<mrc::channel::Status(T&&)> m_on_next;
    std::function<void()> m_on_complete;
};

}  // namespace mrc::node
