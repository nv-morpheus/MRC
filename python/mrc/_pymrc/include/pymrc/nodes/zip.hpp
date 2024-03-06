/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/edge_adapter.hpp"
#include "pymrc/export.h"

#include "mrc/node/operators/zip.hpp"

namespace mrc::pymrc {

template <typename KeyT, typename InputT, typename OutputT = std::vector<InputT>>
class PYMRC_EXPORT PythonDynamicZipComponent : public node::DynamicZipComponent<KeyT, InputT, OutputT>,
                                               public AutoRegSinkAdapter<InputT>,
                                               public AutoRegSourceAdapter<OutputT>
{
    using base_t = node::DynamicZipComponent<KeyT, InputT, OutputT>;

  public:
    using base_t::base_t;
};

}  // namespace mrc::pymrc
