/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

namespace mrc::internal::system {

class System;

/**
 * @brief SystemProvider is a base class designed provide access to the System object.
 *
 * This is on of the most common base classes in the internal library. It was found that many classes needed some
 * information from System, either the Options or the Partition information and commonly both. This base class avoids
 * the repeatitive of taking ownership of a shared_ptr<System> and providing const access to that object.
 *
 * Classes like HostPartitionProvider and PartitionProvider extend SystemProvider by adding other common information
 * needing to be propagated.
 */
class SystemProvider
{
  public:
    SystemProvider(std::shared_ptr<const System> system);
    virtual ~SystemProvider() = default;

    const System& system() const;

  private:
    std::shared_ptr<const System> m_system;
};

}  // namespace mrc::internal::system
