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

#include "mrc/channel/forward.hpp"
#include "mrc/core/userspace_threads.hpp"
#include "mrc/options/resources.hpp"
#include "mrc/options/topology.hpp"

namespace mrc {

// template<class T>
// using blocking_queue = boost::fibers::buffered_channel<T>;

// Typedefs
template <typename T>
using Promise = userspace_threads::promise<T>;  // NOLINT(readability-identifier-naming)

template <typename T>
using Future = userspace_threads::future<T>;  // NOLINT(readability-identifier-naming)

template <typename T>
using SharedFuture = userspace_threads::shared_future<T>;  // NOLINT(readability-identifier-naming)

using Mutex = userspace_threads::mutex;  // NOLINT(readability-identifier-naming)

using CondV = userspace_threads::cv;  // NOLINT(readability-identifier-naming)

using MachineID  = std::uint64_t;  // NOLINT(readability-identifier-naming)
using InstanceID = std::uint64_t;  // NOLINT(readability-identifier-naming)
using TagID      = std::uint64_t;  // NOLINT(readability-identifier-naming)

using NodeID   = std::uint32_t;  // NOLINT(readability-identifier-naming)
using ObjectID = std::uint32_t;  // NOLINT(readability-identifier-naming)

template <typename T>
using Handle = std::shared_ptr<T>;  // NOLINT(readability-identifier-naming)

using SegmentName    = std::string;    // NOLINT(readability-identifier-naming)
using SegmentID      = std::uint16_t;  // NOLINT(readability-identifier-naming)
using SegmentRank    = std::uint16_t;  // NOLINT(readability-identifier-naming)
using SegmentAddress = std::uint32_t;  // NOLINT(readability-identifier-naming) // id + rank

using PortName    = std::string;    // NOLINT(readability-identifier-naming)
using PortID      = std::uint16_t;  // NOLINT(readability-identifier-naming)
using PortGroup   = std::uint32_t;  // NOLINT(readability-identifier-naming)  // port + group_id
using PortAddress = std::uint64_t;  // NOLINT(readability-identifier-naming)  // id + rank + port

using CpuID = std::uint32_t;  // NOLINT(readability-identifier-naming)
using GpuID = std::uint32_t;  // NOLINT(readability-identifier-naming)

using ResourceGroupID = std::size_t;  // NOLINT(readability-identifier-naming)

using Tags = std::vector<SegmentAddress>;  // NOLINT

}  // namespace mrc
