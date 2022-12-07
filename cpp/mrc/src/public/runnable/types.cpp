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

#include "mrc/runnable/types.hpp"

#include <glog/logging.h>

#include <ostream>

namespace mrc::runnable {

std::string engine_type_string(const EngineType& engine_type)
{
    switch (engine_type)
    {
    case EngineType::Fiber:
        return "fiber";
        break;
    case EngineType::Thread:
        return "thread";
        break;
    case EngineType::Process:
        return "process";
        break;
    default:
        LOG(FATAL) << "unknown EngineType";
    }
    return "error";
}

}  // namespace mrc::runnable
