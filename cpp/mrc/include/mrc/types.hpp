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

// Suppress naming conventions in this file to allow matching std and boost libraries
// NOLINTBEGIN(readability-identifier-naming)

namespace mrc {

// Typedefs
template <typename T>
using Promise = userspace_threads::promise<T>;

template <typename T>
using SharedPromise = userspace_threads::shared_promise<T>;

template <typename T>
using Future = userspace_threads::future<T>;

template <typename T>
using SharedFuture = userspace_threads::shared_future<T>;

template <typename SignatureT>
using PackagedTask = userspace_threads::packaged_task<SignatureT>;

using Mutex = userspace_threads::mutex;

using RecursiveMutex = userspace_threads::recursive_mutex;

using CondV = userspace_threads::cv;

using CondVarAny = userspace_threads::cv_any;

using MachineID  = std::uint64_t;
using InstanceID = std::uint64_t;
using TagID      = std::uint64_t;

using NodeID   = std::uint32_t;
using ObjectID = std::uint32_t;

template <typename T>
using Handle = std::shared_ptr<T>;

using SegmentName    = std::string;
using SegmentID      = std::uint16_t;
using SegmentRank    = std::uint16_t;
using SegmentAddress = std::uint32_t;  // id + rank

using PortName    = std::string;
using PortID      = std::uint16_t;
using PortGroup   = std::uint32_t;  // port + group_id
using PortAddress = std::uint64_t;  // id + rank + port

using CpuID = std::uint32_t;
using GpuID = std::uint32_t;

using ResourceGroupID = std::size_t;

using Tags = std::vector<SegmentAddress>;  // NOLINT

// The following lists ID types which are used in MRC. Each ID represents a unique instance in the system. Most of these
// IDs are assigned by the control plane when the instances are created
// All types are 16 bit because they are assigned in increasing order by the control plane and to allow IDs to be
// combined up to 4x for the addresses

using ExecutorID2   = std::uint16_t;  // 16 bit assigned ID for each executor/machine
using PartitionID2  = std::uint16_t;  // 16 bit assigned ID for each worker/machine/partition/connection
using PipelineID2   = std::uint16_t;  // 16 bit assigned ID for each pipelineInstance/pipeline
using SegmentHash2  = std::uint16_t;  // 16 bit hash of segment name
using SegmentID2    = std::uint16_t;  // 16 bit assigned ID for each segmentInstance/segment
using ManifoldID2   = std::uint16_t;  // 16 bit assigned ID for each manifoldInstance/manifold
using ManifoldHash2 = std::uint16_t;  // 16 bit hash of the manifold's port name
using PortID2       = std::uint16_t;  // 16 bit name hash for each port

// The following lists Address types which are used in MRC. Each address represents a unique instance in the system but
// also contains information about the parent resources which the instance belongs to. Combining IDs goes from Parent
// (High bits) -> Child (Low bits). Minimum size for addresses is 32 bits to interface with gRPC better. Unused bits
// should always be 0.

// 16 bit unused + 16 bit ExecutorID
using ExecutorAddress2 = std::uint32_t;
// 16 bit unused + 16 bit PartitionID
using PartitionAddress2 = std::uint32_t;
// 16 bit ExecutorID + 16 bit PipelineID
using PipelineAddress2 = std::uint32_t;
// 16 bit ExecutorID + 16 bit PipelineID + 16 bit SegmentHash + 16 bit SegmentID
using SegmentAddress2 = std::uint64_t;
// 16 bit ExecutorID + 16 bit PipelineID + 16 bit ManifoldHash2 + 16 bit ManifoldID
using ManifoldAddress2 = std::uint64_t;
// 16 bit ExecutorID + 16 bit PipelineID + 16 bit SegmentID + 16 bit PortID
using PortAddress2 = std::uint64_t;

// NOLINTEND(readability-identifier-naming)

}  // namespace mrc
