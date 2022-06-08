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

#include <srf/forward.hpp>

// todo
// - revert adding tasks directly to operators
// - create a OpTask where each node owns a taskflow for its operations
// - auto op_graph = node.op_graph(); // op_graph owns the task handle for the data node
// - OpTask owns the task and the unique_ptr<Operator<T>>

#include <srf/core/context.hpp>
#include <srf/core/executor.hpp>
#include <srf/options/options.hpp>
#include <srf/pipeline/pipeline.hpp>
#include <srf/segment/builder.hpp>
#include <srf/segment/definition.hpp>
#include <srf/segment/egress_ports.hpp>
#include <srf/segment/ingress_ports.hpp>

namespace srf {

inline std::unique_ptr<pipeline::Pipeline> make_pipeline()
{
    return pipeline::Pipeline::create();
}

}  // namespace srf
