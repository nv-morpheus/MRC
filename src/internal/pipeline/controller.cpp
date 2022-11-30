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

#include "internal/pipeline/controller.hpp"

#include "internal/pipeline/types.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <rxcpp/rx.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

namespace mrc::internal::pipeline {

Controller::Controller(std::unique_ptr<Instance> pipeline) : m_pipeline(std::move(pipeline))
{
    CHECK(m_pipeline);
    m_pipeline->service_start();
}

void Controller::on_data(ControlMessage&& message)
{
    auto& ctx = mrc::runnable::Context::get_runtime_context();
    DCHECK_EQ(ctx.size(), 1);

    switch (message.type)
    {
    case ControlMessageType::Update:
        try
        {
            update(std::move(message.addresses));
        } catch (...)
        {
            LOG(ERROR) << "exception caught while performing update - this is fatal - issuing kill";
            kill();
            std::rethrow_exception(std::current_exception());
        }
        break;
    case ControlMessageType::Stop:
        stop();
        break;
    case ControlMessageType::Kill:
        kill();
        break;
    default:
        LOG(FATAL) << "Unhandled ControlMessageType";
    }
}

void Controller::stop()
{
    m_pipeline->service_stop();
}

void Controller::kill()
{
    m_pipeline->service_kill();
}

void Controller::await_on_pipeline() const
{
    m_pipeline->service_await_join();
}

void Controller::update(SegmentAddresses&& new_segments_map)
{
    VLOG(10) << info() << ": starting update";

    auto cur_segments = extract_keys(m_current_segments);
    auto new_segments = extract_keys(new_segments_map);

    // set of segments to remove
    std::set<SegmentAddress> create_segments;
    std::set_difference(new_segments.begin(),
                        new_segments.end(),
                        cur_segments.begin(),
                        cur_segments.end(),
                        std::inserter(create_segments, create_segments.end()));
    DVLOG(10) << info() << create_segments.size() << " segments will be created";

    // set of segments to remove
    std::set<SegmentAddress> remove_segments;
    std::set_difference(cur_segments.begin(),
                        cur_segments.end(),
                        new_segments.begin(),
                        new_segments.end(),
                        std::inserter(remove_segments, remove_segments.end()));
    DVLOG(10) << info() << remove_segments.size() << " segments marked for removal";

    // construct new segments and attach to manifold
    for (const auto& address : create_segments)
    {
        auto partition_id = new_segments_map.at(address);
        DVLOG(10) << info() << ": create segment for address " << ::mrc::segment::info(address)
                  << " on resource partition: " << partition_id;
        m_pipeline->create_segment(address, partition_id);
    }

    // detach from manifold or stop old segments
    for (const auto& address : remove_segments)
    {
        DVLOG(10) << info() << ": stop segment for address " << ::mrc::segment::info(address);
        m_pipeline->stop_segment(address);
    }

    // m_pipeline->manifold_update_inputs();

    m_pipeline->update();

    // when ready issue update
    // this should start all segments
    // m_pipeline->update();

    // update manifolds

    // await old segments

    // update current segments
    m_current_segments = std::move(new_segments_map);

    VLOG(10) << info() << ": update complete";
}

void Controller::did_complete()
{
    VLOG(10) << info() << ": received shutdown notification - channel closed no new assigments will be issued";
}

const std::string& Controller::info()
{
    static std::string str = "pipeline::controller";
    return str;
}

}  // namespace mrc::internal::pipeline
