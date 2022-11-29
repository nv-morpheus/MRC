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

#include "mrc/metrics/registry.hpp"

#include "mrc/metrics/counter.hpp"

#include <glog/logging.h>
#include <prometheus/client_metric.h>
#include <prometheus/counter.h>
#include <prometheus/family.h>
#include <prometheus/registry.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mrc::metrics {

Registry::Registry() :
  m_registry(std::make_shared<prometheus::Registry>()),
  m_throughput_counters(prometheus::BuildCounter()
                            .Name("mrc_throughput_counters")
                            .Help("number of data elements passing thru a given pipeline object")
                            .Register(*m_registry))
{}

Counter Registry::make_counter(std::string name, std::map<std::string, std::string> labels)
{
    auto& family  = prometheus::BuildCounter().Name(std::move(name)).Register(*m_registry);
    auto& counter = family.Add(std::move(labels));
    return Counter(&counter);
}

Counter Registry::make_throughput_counter(std::string name)
{
    auto& counter = m_throughput_counters.Add({{"name", name}});
    return Counter(&counter);
}

std::vector<CounterReport> Registry::collect_throughput_counters() const
{
    std::vector<CounterReport> report;
    auto collected = m_throughput_counters.Collect();
    CHECK_EQ(collected.size(), 1);
    for (auto& metric : collected[0].metric)
    {
        report.emplace_back(metric.label.at(0).value, metric.counter.value);
    }
    return report;
}

}  // namespace mrc::metrics
