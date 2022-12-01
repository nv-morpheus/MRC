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

#include "mrc/cuda/sync.hpp"

#include "mrc/cuda/common.hpp"  // IWYU pragma: associated
#include "mrc/types.hpp"        // for Promise, Future

#include <boost/fiber/future/promise.hpp>

namespace mrc {

static void enqueue_stream_event_callback(void* user_data)
{
    auto* promise = static_cast<Promise<void>*>(user_data);
    promise->set_value();
    delete promise;
}

Future<void> enqueue_stream_sync_event(cudaStream_t stream)
{
    auto* promise = new Promise<void>;
    auto future   = promise->get_future();
    MRC_CHECK_CUDA(cudaLaunchHostFunc(stream, enqueue_stream_event_callback, promise));
    return future;
}

}  // namespace mrc
