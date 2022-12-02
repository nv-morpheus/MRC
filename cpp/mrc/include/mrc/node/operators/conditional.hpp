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
#include "mrc/node/operators/router.hpp"

namespace mrc::node {

template <typename T, typename CaseT>
class Conditional : public Operator<T>, public RouterBase<CaseT, T>
{
  public:
    Conditional(std::function<CaseT(const T&)> predicate) : m_predicate(std::move(predicate)) {}

  private:
    inline channel::Status on_next(T&& data) final
    {
        return this->channel_for_key(m_predicate(data)).await_write(std::move(data));
    }

    // Operator::on_release
    void on_complete() final
    {
        RouterBase<CaseT, T>::release_sources();
    }

    std::function<CaseT(const T&)> m_predicate;
};

}  // namespace mrc::node
