/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/utils/macros.hpp"

#include <functional>
#include <memory>

namespace mrc {
class Options;
}  // namespace mrc

namespace mrc::pipeline {

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
    virtual ~ISystem() = default;
    DELETE_COPYABILITY(ISystem);

    virtual const Options& options() const = 0;

    virtual void add_thread_initializer(std::function<void()> initializer_fn) = 0;
    virtual void add_thread_finalizer(std::function<void()> finalizer_fn)     = 0;

  protected:
    ISystem() = default;
};

}  // namespace mrc::pipeline

namespace mrc {
std::unique_ptr<pipeline::ISystem> make_system(std::shared_ptr<Options> options = nullptr);
}
