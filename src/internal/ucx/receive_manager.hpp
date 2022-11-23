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

#include "internal/ucx/common.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/types.hpp"

#include <ucp/api/ucp_def.h>  // for ucp_tag_t, ucp_tag_message_h, ucp_tag_recv_info_t

#include <memory>  // for enable_shared_from_this

namespace mrc::internal::ucx {

class TaggedReceiveManager : public std::enable_shared_from_this<TaggedReceiveManager>
{
    static constexpr ucp_tag_t MATCH_ALL_BITS = 0x0000000000000000;  // NOLINT

  public:
    TaggedReceiveManager(Handle<Worker> worker, ucp_tag_t tag = 0, ucp_tag_t task_mask = MATCH_ALL_BITS);
    virtual ~TaggedReceiveManager();

    WorkerAddress local_address();

    void start();
    void stop();
    void join();

    bool is_running() const
    {
        return m_running;
    }

    Worker& worker();

  private:
    void progress_engine();

    virtual void on_tagged_msg(ucp_tag_message_h, const ucp_tag_recv_info_t&) = 0;

    Handle<Worker> m_worker;
    ucp_tag_t m_tag;
    ucp_tag_t m_tag_mask;

    Future<void> m_shutdown_complete;
    mutable Mutex m_mutex;
    bool m_running;
};

}  // namespace mrc::internal::ucx
