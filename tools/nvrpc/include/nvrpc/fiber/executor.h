/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvrpc/executor.h>

#include <boost/fiber/all.hpp>

namespace nvrpc {
// todo: the derviation of FiberExecutor required making the variables of Executor protected
// instead of private.  work should be to clean up the interface and establish better
// inheritance properites
class FiberExecutor : public Executor
{
    using Executor::Executor;

    /*
        void ProgressEngine(int thread_id) override
        {
            //::srf::async::shared_work_pool<WorkPoolID>();
            ::boost::fibers::use_scheduling_algorithm<::boost::fibers::algo::shared_work>();
            bool ok;
            void* tag;
            auto myCQ = m_ServerCompletionQueues[thread_id].get();
            m_Running = true;

            while(myCQ->Next(&tag, &ok))
            {
                ::boost::fibers::fiber([this, tag, ok]() mutable {
                    DVLOG(3) << "execution fiber " << boost::this_fiber::get_id() << " running on thread "
                             << std::this_thread::get_id();
                    auto ctx = IContext::Detag(tag);
                    if(!RunContext(ctx, ok))
                    {
                        if(m_Running)
                        {
                            ResetContext(ctx);
                        }
                    }
                }).detach();
            }
        }
    */
};

}  // namespace nvrpc
