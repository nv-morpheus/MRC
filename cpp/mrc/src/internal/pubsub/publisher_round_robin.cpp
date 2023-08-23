/*
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

#include "internal/pubsub/publisher_round_robin.hpp"

#include "internal/data_plane/client.hpp"
#include "internal/resources/partition_resources.hpp"
#include "internal/runnable/runnable_resources.hpp"

#include "mrc/core/task_queue.hpp"

#include <glog/logging.h>

#include <atomic>
#include <ostream>
#include <utility>

namespace mrc::pubsub {

void PublisherRoundRobin::on_update()
{
    m_next = this->tagged_endpoints().cbegin();
}

void PublisherRoundRobin::apply_policy(rxcpp::subscriber<data_plane::LocalDescriptorMessage>& sub,
                                       runtime::LocalDescriptorHandle descriptor_handle)
{
    DCHECK(this->resources().runnable().main().caller_on_same_thread());

    // todo(cpp20) - blocking in this method is blocking in the parent's policy engine
    // in order to avoid a deadlock, we need a stop token here in the event that we have a remote descriptor that is
    // trying to be written, and there are no subscribers/tagged_instances and the publisher is being dropped
    // the await_join on the policy engine's runner will never complete because the progress loop cannot exit
    // while (this->tagged_instances().empty())
    // {
    //     // await subscribers
    //     boost::this_fiber::yield();
    // }

    if (tagged_instances().empty())
    {
        LOG_EVERY_N(WARNING, 1000) << "publisher dropping object because no subscribers are active";  // NOLINT
    }

    sub.on_next(data_plane::LocalDescriptorMessage(std::move(descriptor_handle), m_next->second, m_next->first));

    if (++m_next == this->tagged_endpoints().cend())
    {
        m_next = this->tagged_endpoints().cbegin();
    }
}

}  // namespace mrc::pubsub
