/**
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

#include "mrc/modules/stream_buffer/stream_buffer_base.hpp"

#include <cstddef>
#include <type_traits>

namespace mrc::modules::stream_buffers {

template <typename DataTypeT, template <typename> class StreamBufferTypeT>
concept IsStreamBuffer = requires {
                             typename StreamBufferTypeT<DataTypeT>;
                             std::is_base_of_v<StreamBufferBase<DataTypeT>, StreamBufferTypeT<DataTypeT>>;
                         };
}  // namespace mrc::modules::stream_buffers
