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

#include "mrc/options/options.hpp"

#include <memory>

namespace mrc::internal::system {

class System;

/**
 * @brief System object
 *
 * Core class that could be used to transfer Topology and Partition information from the MRC runtime.
 *
 * Currently, this is only an opaque handle for constructing a system::IResource.
 */
class ISystem
{
  public:
    ISystem(std::shared_ptr<Options> options);
    virtual ~ISystem() = 0;

  private:
    std::shared_ptr<System> m_impl;
    friend System;
};

}  // namespace mrc::internal::system
