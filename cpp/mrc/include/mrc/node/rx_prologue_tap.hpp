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

#include <rxcpp/rx.hpp>

#include <functional>
#include <vector>

namespace mrc::node {

// set of tap operators that are inserted into the rx chain and executor just after messages are read.
template <typename T>
class RxPrologueTap
{
  public:
    void add_prologue_tap(std::function<void(const T&)> tap_fn)
    {
        m_taps.push_back(tap_fn);
    }

  protected:
    rxcpp::observable<T> apply_prologue_taps(rxcpp::observable<T> observable)
    {
        rxcpp::observable<T> obs = observable;
        for (auto& tap : m_taps)
        {
            obs = obs.tap(tap);
        }
        return obs;
    }

  private:
    std::vector<std::function<void(const T&)>> m_taps;
};

}  // namespace mrc::node
