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

#include "mrc/core/executor.hpp"

#include "mrc/engine/executor/iexecutor.hpp"
#include "mrc/engine/system/iresources.hpp"
#include "mrc/options/options.hpp"

#include <utility>  // for move

namespace mrc {

Executor::Executor() : internal::executor::IExecutor(std::make_shared<Options>()) {}
Executor::Executor(std::shared_ptr<Options> options) : internal::executor::IExecutor(std::move(options)) {}
Executor::Executor(std::unique_ptr<internal::system::IResources> resources) :
  internal::executor::IExecutor(std::move(resources))
{}

}  // namespace mrc
