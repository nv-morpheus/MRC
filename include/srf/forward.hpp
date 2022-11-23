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

namespace mrc {

class Executor;
class ExecutorBase;

template <typename T>
class NetworkDeserializer;

class Options;

class Runtime;

class Placement;
class PlacementGroup;

namespace pipeline {
class Pipeline;
}  // namespace pipeline

class PipelineConfiguration;

class SegmentResources;

class PipelineInstanceResources;

class PipelineConfiguration;
class SegmentConfiguration;

class PipelineAssignment;
class SegmentAssignment;

class Cpuset;
class NumaSet;

class Options;
class FiberPoolOptions;
class ResourceOptions;
class MemoryPoolOptions;
class TopologyOptions;

}  // namespace mrc
