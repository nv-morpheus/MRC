/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "internal/runtime/segments_manager.hpp"

#include "internal/control_plane/state/root_state.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/runtime/partition_runtime.hpp"
#include "internal/runtime/pipelines_manager.hpp"
#include "internal/runtime/resource_manager_base.hpp"
#include "internal/runtime/runtime_provider.hpp"
#include "internal/segment/segment_definition.hpp"
#include "internal/system/partition.hpp"
#include "internal/ucx/worker.hpp"
#include "internal/utils/ranges.hpp"

#include "mrc/core/addresses.hpp"
#include "mrc/core/async_service.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/protos/architect.pb.h"
#include "mrc/protos/architect_state.pb.h"
#include "mrc/types.hpp"
#include "mrc/utils/string_utils.hpp"

#include <glog/logging.h>
#include <google/protobuf/util/message_differencer.h>
#include <rxcpp/rx.hpp>

#include <chrono>
#include <memory>

namespace mrc::runtime {

SegmentsManager::SegmentsManager(runtime::IInternalPartitionRuntimeProvider& runtime, size_t partition_id) :
  AsyncService(MRC_CONCAT_STR("SegmentsManager[" << partition_id << "]")),
  InternalPartitionRuntimeProvider(runtime)
{}

SegmentsManager::~SegmentsManager()
{
    AsyncService::call_in_destructor();
}

void SegmentsManager::do_service_start(std::stop_token stop_token)
{
    Promise<void> completed_promise;

    std::stop_callback stop_callback(stop_token, [&completed_promise]() {
        completed_promise.set_value();
    });

    this->mark_started();

    completed_promise.get_future().get();
}

void SegmentsManager::sync_state(const control_plane::state::Worker& worker)
{
    // Check for assignments
    auto cur_segments = extract_keys(m_instances);
    auto new_segments = extract_keys(worker.assigned_segments());

    // auto [create_segments, remove_segments] = compare_difference(cur_segments, new_segments);

    // set of segments to remove
    std::set<SegmentAddress> create_segments;
    std::set_difference(new_segments.begin(),
                        new_segments.end(),
                        cur_segments.begin(),
                        cur_segments.end(),
                        std::inserter(create_segments, create_segments.end()));
    DVLOG(10) << create_segments.size() << " segments will be created";

    // set of segments to remove
    std::set<SegmentAddress> remove_segments;
    std::set_difference(cur_segments.begin(),
                        cur_segments.end(),
                        new_segments.begin(),
                        new_segments.end(),
                        std::inserter(remove_segments, remove_segments.end()));
    DVLOG(10) << remove_segments.size() << " segments marked for removal";

    // construct new segments and attach to manifold
    for (const auto& address : create_segments)
    {
        // auto partition_id = new_segments_map.at(address);
        // DVLOG(10) << info() << ": create segment for address " << ::mrc::segment::info(address)
        //           << " on resource partition: " << partition_id;
        this->create_segment(worker.assigned_segments().at(address));
    }

    // detach from manifold or stop old segments
    for (const auto& address : remove_segments)
    {
        // DVLOG(10) << info() << ": stop segment for address " << ::mrc::segment::info(address);
        this->erase_segment(address);
    }
}

// void SegmentsManager::process_state_update(mrc::control_plane::state::Worker& instance)
// {
//     switch (worker.state().requested_status())
//     {
//     case control_plane::state::ResourceRequestedStatus::Initialized:
//     case control_plane::state::ResourceRequestedStatus::Created: {
//         // Activate our worker
//         protos::RegisterWorkersResponse resp;
//         resp.set_machine_id(0);
//         resp.add_instance_ids(m_worker_id);

//         // Need to activate our worker
//         m_runtime.control_plane().await_unary<protos::Ack>(protos::ClientUnaryActivateStream, std::move(resp));

//         // Indicate this is now live
//         this->mark_started();
//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Completed: {
//         // Check for assignments
//         auto cur_segments = extract_keys(m_instances);
//         auto new_segments = extract_keys(worker.assigned_segments());

//         // auto [create_segments, remove_segments] = compare_difference(cur_segments, new_segments);

//         // set of segments to remove
//         std::set<SegmentAddress> create_segments;
//         std::set_difference(new_segments.begin(),
//                             new_segments.end(),
//                             cur_segments.begin(),
//                             cur_segments.end(),
//                             std::inserter(create_segments, create_segments.end()));
//         DVLOG(10) << create_segments.size() << " segments will be created";

//         // set of segments to remove
//         std::set<SegmentAddress> remove_segments;
//         std::set_difference(cur_segments.begin(),
//                             cur_segments.end(),
//                             new_segments.begin(),
//                             new_segments.end(),
//                             std::inserter(remove_segments, remove_segments.end()));
//         DVLOG(10) << remove_segments.size() << " segments marked for removal";

//         // construct new segments and attach to manifold
//         for (const auto& address : create_segments)
//         {
//             // auto partition_id = new_segments_map.at(address);
//             // DVLOG(10) << info() << ": create segment for address " << ::mrc::segment::info(address)
//             //           << " on resource partition: " << partition_id;
//             this->create_segment(worker.assigned_segments().at(address));
//         }

//         // detach from manifold or stop old segments
//         for (const auto& address : remove_segments)
//         {
//             // DVLOG(10) << info() << ": stop segment for address " << ::mrc::segment::info(address);
//             this->erase_segment(address);
//         }

//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Stopped: {
//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Destroyed: {
//         break;
//     }
//     case control_plane::state::ResourceRequestedStatus::Unknown:
//     default: {
//         CHECK(false) << "Unknown worker state: " << static_cast<int>(worker.state().requested_status());
//     }
//     }
// }

void SegmentsManager::create_segment(const mrc::control_plane::state::SegmentInstance& instance_state)
{
    // Get a reference to the pipeline we are creating the segment in
    auto& pipeline_def = this->runtime().pipelines_manager().get_definition(instance_state.pipeline_definition().id());

    auto [id, rank] = segment_address_decode(instance_state.address());
    auto definition = pipeline_def.find_segment(id);

    auto [added_iterator, did_add] = m_instances.emplace(
        instance_state.address(),
        std::make_unique<segment::SegmentInstance>(*this,
                                                   definition,
                                                   instance_state.address(),
                                                   instance_state.pipeline_instance().id()));

    // Now start as a child service
    this->child_service_start(*added_iterator->second);

    // // Create the resource on the correct runnable
    // m_runtime.resources()
    //     .runnable()
    //     .main()
    //     .enqueue([this, instance_state] {
    //         // Make sure the segment is live before continuing
    //         // added_iterator->second->service_await_live();

    //         // for (const auto& name : definition->egress_port_names())
    //         // {
    //         //     VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for egress port " << name;
    //         //     std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
    //         //     if (!manifold)
    //         //     {
    //         //         VLOG(10) << ::mrc::segment::info(address) << " creating manifold for egress port " << name;
    //         //         manifold          = segment->create_manifold(name);
    //         //         m_manifolds[name] = manifold;
    //         //     }
    //         //     segment->attach_manifold(manifold);
    //         // }

    //         // for (const auto& name : definition->ingress_port_names())
    //         // {
    //         //     VLOG(10) << ::mrc::segment::info(address) << " configuring manifold for ingress port " << name;
    //         //     std::shared_ptr<manifold::Interface> manifold = get_manifold(name);
    //         //     if (!manifold)
    //         //     {
    //         //         VLOG(10) << ::mrc::segment::info(address) << " creating manifold for ingress port " << name;
    //         //         manifold          = segment->create_manifold(name);
    //         //         m_manifolds[name] = manifold;
    //         //     }
    //         //     segment->attach_manifold(manifold);
    //         // }

    //         // m_segments[address] = std::move(segment);
    //     })
    //     .get();

    DVLOG(10) << "Manager created SegmentInstance: " << instance_state.id() << "/" << instance_state.name();
}

void SegmentsManager::erase_segment(SegmentAddress address)
{
    CHECK(m_instances[address]->service_await_join(std::chrono::milliseconds(100)))
        << "SegmentInstance[" << address << "] did not shut down quickly";

    CHECK_EQ(m_instances.erase(address), 1) << "SegmentInstance[" << address << "] could not be removed";
}

}  // namespace mrc::runtime
