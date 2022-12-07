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

#include "mrc/types.hpp"  // for Future

#include <cuda_runtime.h>

namespace mrc {

/**
 * @brief enqueue on to a stream an awaitable promise that will be fulfilled
 *        when all preceding kernels in the stream have completed.
 *
 * @param [in] stream
 *
 * @return Future<void>
 *
 * @note - the current implementation uses cudaLaunchHostFunc; however, an
 * alternative poll / fiber yield maybe added in the future.
 */
Future<void> enqueue_stream_sync_event(cudaStream_t stream);

}  // namespace mrc
