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

#include "mrc/metrics/counter.hpp"

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// IWYU pragma: no_include "prometheus/family.h"

namespace prometheus {
class Registry;

template <typename T>
class Family;  // IWYU pragma: keep

}  // namespace prometheus

namespace mrc::metrics {

struct CounterReport
{
    CounterReport(std::string n, std::size_t c) : name(std::move(n)), count(c) {}
    std::string name;
    std::size_t count;
};

class Registry
{
  public:
    Registry();

    Counter make_counter(std::string name, std::map<std::string, std::string> labels);
    Counter make_throughput_counter(std::string);

    std::vector<CounterReport> collect_throughput_counters() const;

  protected:
  private:
    std::shared_ptr<prometheus::Registry> m_registry;
    prometheus::Family<prometheus::Counter>& m_throughput_counters;
};

}  // namespace mrc::metrics
