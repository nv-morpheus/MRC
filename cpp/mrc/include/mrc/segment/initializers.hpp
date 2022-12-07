/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mrc/engine/segment/ibuilder.hpp"
#include "mrc/segment/forward.hpp"  // IWYU pragma: export
#include "mrc/types.hpp"

#include <functional>
#include <memory>

namespace mrc::internal::segment {
class IBuilder;
}

namespace mrc::segment {

using segment_initializer_fn_t = std::function<void(Builder&)>;
using egress_initializer_t     = std::function<std::shared_ptr<EgressPortBase>(const SegmentAddress&)>;
using ingress_initializer_t    = std::function<std::shared_ptr<IngressPortBase>(const SegmentAddress&)>;
using backend_initializer_fn_t = std::function<void(internal::segment::IBuilder&)>;

}  // namespace mrc::segment
