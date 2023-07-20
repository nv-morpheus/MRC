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

#pragma once

#include "mrc/core/bitmap.hpp"
#include "mrc/pipeline/system.hpp"
#include "mrc/utils/macros.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace mrc {
class Options;
}  // namespace mrc

namespace mrc::system {

class Partitions;
class Topology;

class SystemDefinition final : public pipeline::ISystem
{
  public:
    SystemDefinition(const Options& options);
    SystemDefinition(std::shared_ptr<Options> options);
    ~SystemDefinition() override;

    static std::unique_ptr<SystemDefinition> unwrap(std::unique_ptr<ISystem> object);

    DELETE_COPYABILITY(SystemDefinition);
    DELETE_MOVEABILITY(SystemDefinition);

    const Options& options() const override;
    const Topology& topology() const;
    const Partitions& partitions() const;

    void add_thread_initializer(std::function<void()> initializer_fn) override;
    void add_thread_finalizer(std::function<void()> finalizer_fn) override;

    const std::vector<std::function<void()>>& thread_initializers() const;
    const std::vector<std::function<void()>>& thread_finalizers() const;

    CpuSet get_current_thread_affinity() const;

  private:
    std::unique_ptr<const Options> m_options;
    std::shared_ptr<Topology> m_topology;
    std::shared_ptr<Partitions> m_partitions;

    std::vector<std::function<void()>> m_thread_initializers;
    std::vector<std::function<void()>> m_thread_finalizers;
};

}  // namespace mrc::system
