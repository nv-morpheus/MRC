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

#include "internal/ucx/receive_manager.hpp"

#include "internal/ucx/common.hpp"
#include "internal/ucx/worker.hpp"

#include "mrc/types.hpp"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>
#include <boost/fiber/policy.hpp>  // for launch, launch::post
#include <ucp/api/ucp.h>           // for ucp_tag_probe_nb, ucp_tag_recv_info

#include <chrono>   // for nanoseconds
#include <cstdint>  // for uint32_t
#include <utility>

namespace mrc::internal::ucx {

TaggedReceiveManager::TaggedReceiveManager(Handle<Worker> worker, ucp_tag_t tag, ucp_tag_t task_mask) :
  m_worker(std::move(worker)),
  m_tag(tag),
  m_tag_mask(task_mask),
  m_running(false)
{}

TaggedReceiveManager::~TaggedReceiveManager() = default;

void TaggedReceiveManager::start()
{
    m_running           = true;
    m_shutdown_complete = boost::fibers::async(::boost::fibers::launch::post, [this] { progress_engine(); });
}

void TaggedReceiveManager::stop()
{
    m_running = false;
}

void TaggedReceiveManager::join()
{
    m_shutdown_complete.get();
}

Worker& TaggedReceiveManager::worker()
{
    return *m_worker;
}
WorkerAddress TaggedReceiveManager::local_address()
{
    return m_worker->address();
}

void TaggedReceiveManager::progress_engine()
{
    ucp_tag_message_h msg;
    ucp_tag_recv_info_t msg_info;
    std::uint32_t backoff = 1;

    while (true)
    {
        for (;;)
        {
            msg = ucp_tag_probe_nb(m_worker->handle(), m_tag, m_tag_mask, 1, &msg_info);
            if (msg != nullptr)
            {
                break;
            }
            while (m_worker->progress() != 0U)
            {
                backoff = 1;
            }
            if (!m_running)
            {
                return;
            }

            if (backoff < 1048576)
            {
                backoff = backoff << 1;
            }
            if (backoff < 32768)
            {
                boost::this_fiber::yield();
            }
            else
            {
                boost::this_fiber::sleep_for(std::chrono::nanoseconds(backoff));
            }
        }

        on_tagged_msg(msg, msg_info);
        backoff = 0;
    }
}

}  // namespace mrc::internal::ucx
