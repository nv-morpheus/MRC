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

#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>

namespace srf::internal::data_plane {

struct Callbacks final
{
    // internal point-to-point
    static void send(void* request, ucs_status_t status, void* user_data);
    static void recv(void* request, ucs_status_t status, const ucp_tag_recv_info_t* msg_info, void* user_data);
};

}  // namespace srf::internal::data_plane
