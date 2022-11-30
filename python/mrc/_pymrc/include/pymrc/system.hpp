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

#include "mrc/engine/system/iresources.hpp"
#include "mrc/engine/system/isystem.hpp"
#include "mrc/options/options.hpp"

#include <memory>

namespace mrc::pymrc {

class System final : public internal::system::ISystem
{
  public:
    System(std::shared_ptr<Options> options);
    ~System() final = default;
};

class SystemResources final : public internal::system::IResources
{
  public:
    SystemResources(std::shared_ptr<System> system);
    ~SystemResources() final = default;

  private:
    void add_gil_initializer();
    void add_gil_finalizer();
};

}  // namespace mrc::pymrc
