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

#include "mrc/channel/forward.hpp"
#include "mrc/core/userspace_threads.hpp"
#include "mrc/options/resources.hpp"
#include "mrc/options/topology.hpp"

namespace mrc {

// Suppress naming conventions in this file to allow matching std and boost libraries
// NOLINTBEGIN(readability-identifier-naming)

// Typedefs
template <typename T>
using Promise = userspace_threads::promise<T>;

template <typename T>
using Future = userspace_threads::future<T>;

template <typename T>
using SharedFuture = userspace_threads::shared_future<T>;

using Mutex = userspace_threads::mutex;

using RecursiveMutex = userspace_threads::recursive_mutex;

using CondV = userspace_threads::cv;

using MachineID  = std::uint64_t;
using InstanceID = std::uint64_t;
using TagID      = std::uint64_t;

template <typename T>
using Handle = std::shared_ptr<T>;

using SegmentID      = std::uint16_t;
using SegmentRank    = std::uint16_t;
using SegmentAddress = std::uint32_t;  // id + rank

using PortName    = std::string;
using PortID      = std::uint16_t;
using PortAddress = std::uint64_t;  // id + rank + port

enum class State
{
    Init   = 0,
    Run    = 1,
    Joined = 2,
    Stop   = 3,
    Kill   = 4
};

// Stage change callback passed to the executor
using on_state_change_fn = std::function<void(State)>;

// NOLINTEND(readability-identifier-naming)

}  // namespace mrc
