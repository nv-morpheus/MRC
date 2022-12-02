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

#include "mrc/runnable/context.hpp"

namespace mrc {

/**
 * @brief Acquire the local context for the current Runnable.
 *
 * Runnable Contexts are available on the thread/fiber that initialized the Runnable. An error will be thrown if trying
 * to acquire a Contexts outside the scope of a Runnable.
 *
 * @return runnable::Context&
 */
runnable::Context& get_current_context()
{
    return runnable::Context::get_runtime_context();
}

}  // namespace mrc
