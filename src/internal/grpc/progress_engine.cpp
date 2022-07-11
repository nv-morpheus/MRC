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

#include "internal/grpc/progress_engine.hpp"

namespace srf::internal::rpc {

ProgressEngine::ProgressEngine(std::shared_ptr<grpc::CompletionQueue> cq) : m_cq(std::move(cq)) {}

void ProgressEngine::data_source(rxcpp::subscriber<ProgressEvent>& s)
{
    ProgressEvent event;
    std::uint64_t backoff = 128;

    DVLOG(10) << "starting progress engine";

    while (s.is_subscribed())
    {
        switch (m_cq->AsyncNext<gpr_timespec>(&event.tag, &event.ok, gpr_time_0(GPR_CLOCK_REALTIME)))
        {
        case grpc::CompletionQueue::NextStatus::GOT_EVENT: {
            backoff = 128;
            DVLOG(20) << "progress engine got event";
            s.on_next(event);
        }
        break;
        case grpc::CompletionQueue::NextStatus::TIMEOUT: {
            if (backoff < 1048576)
            {
                backoff = (backoff << 1);
            }
            boost::this_fiber::sleep_for(std::chrono::microseconds(backoff));
        }
        break;
        case grpc::CompletionQueue::NextStatus::SHUTDOWN: {
            DVLOG(10) << "progress engine complete";
            return;
        }
        }
    }
}

void ProgressEngine::on_stop(const rxcpp::subscription& subscription) {}

}  // namespace srf::internal::rpc
