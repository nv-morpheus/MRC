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
#include "mrc/utils/macros.hpp"

#include <memory>

namespace mrc {
class Options;
}  // namespace mrc

namespace mrc::system {

class ISystem;
class Partitions;
class Topology;

class System final
{
    System(std::shared_ptr<Options> options);

  public:
    static std::shared_ptr<System> create(std::shared_ptr<Options> options);
    static std::shared_ptr<System> unwrap(const ISystem& system);

    ~System() = default;

    DELETE_COPYABILITY(System);
    DELETE_MOVEABILITY(System);

    const Options& options() const;
    const Topology& topology() const;
    const Partitions& partitions() const;

    CpuSet get_current_thread_affinity() const;

  private:
    std::shared_ptr<Options> m_options;
    std::shared_ptr<Topology> m_topology;
    std::shared_ptr<Partitions> m_partitions;
};

std::shared_ptr<System> make_system(std::shared_ptr<Options> options);

}  // namespace mrc::system
