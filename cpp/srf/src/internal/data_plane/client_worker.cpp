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

#include "internal/data_plane/client_worker.hpp"

#include "internal/ucx/worker.hpp"

#include <ucp/api/ucp_compat.h>
#include <boost/fiber/operations.hpp>

namespace srf::internal::data_plane {

void DataPlaneClientWorker::on_data(void*&& data)
{
    while (ucp_request_is_completed(data) == 0)
    {
        if (m_worker->progress() != 0U)
        {
            continue;
        }
        boost::this_fiber::yield();
    }
    ucp_request_release(data);
}

}  // namespace srf::internal::data_plane
