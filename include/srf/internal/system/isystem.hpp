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

#include <srf/options/options.hpp>

#include <functional>
#include <memory>

namespace srf::internal::system {

class System;

class ISystem
{
  public:
    ISystem(std::shared_ptr<Options> options);
    virtual ~ISystem() = 0;

  protected:
    // void add_thread_initializer(std::function<void()> initializer_fn);
    // void add_thread_finalizer(std::function<void()> finalizer_fn);

  private:
    std::shared_ptr<System> m_impl;
    friend System;
};

}  // namespace srf::internal::system
