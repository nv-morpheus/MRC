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

#include "mrc/manifold/manifold.hpp"

#include "mrc/channel/status.hpp"
#include "mrc/core/utils.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/types.hpp"
#include "mrc/utils/ranges.hpp"

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace mrc::manifold {

// void ManifoldTaggerBase::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
// {
//     // Call the base and then update the list of available tags
//     ManifoldNodeBase::add_output(address, output_sink);
// }

InstanceID ManifoldTaggerBase2::get_next_tag()
{
    return m_current_policy.get_next_tag();
}

bool ManifoldTaggerBase2::has_connections() const
{
    return m_current_policy.has_connections();
}

// void ManifoldTaggerBase::do_update_policy(const ManifoldPolicy& policy)
// {
//     // Clear the list of existing outputs
//     this->drop_outputs();

//     // for (const auto& info : policy.outputs)
//     // {
//     //     CHECK(info.edge != nullptr) << "Cannot set an empty edge for SegmentAddress: " << info.address;
//     //     mrc::make_edge(this->get_output(info.address), *info.edge);
//     // }
// }

void ManifoldTaggerBase2::update_policy(ManifoldPolicy&& policy)
{
    boost::fibers::packaged_task<void()> update_task([this, policy = std::move(policy)]() {
        auto next_keys = extract_keys(policy.outputs);

        auto [to_add, to_remove] = compare_difference(this->writable_edge_keys(),
                                                      std::vector<InstanceID>{next_keys.begin(), next_keys.end()});

        for (const auto& key : to_remove)
        {
            this->release_writable_edge(key);
        }

        for (const auto& key : to_add)
        {
            const auto& output = policy.outputs.at(key);

            this->add_output(key, output.is_local, output.edge);
        }

        // Set a flag to indicate if we have at least one local input connection and one (local or remote) output
        // connection
        m_has_local_input = policy.local_input_count() > 0 && !policy.outputs.empty();

        m_current_policy = std::move(policy);
    });

    auto update_future = update_task.get_future();

    CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

    // Before continuing, wait for the update to be processed
    update_future.get();
}

void ManifoldTaggerBase2::run(runnable::Context& ctx)
{
    std::uint64_t backoff = 128;

    while (this->state() == State::Run)
    {
        // if we are rank 0, check for updates
        if (ctx.rank() == 0)
        {
            channel::Status update_status;
            boost::fibers::packaged_task<void()> next_update;

            while ((update_status = m_updates.try_read(next_update)) == channel::Status::success)
            {
                // Run the next update
                next_update();
            }
        }

        // Barrier to sync threads
        ctx.barrier();

        // Default is timeout for when we dont have connections
        channel::Status status = channel::Status::timeout;

        if (m_has_local_input)
        {
            // Try and process a message. Use the return value to alter the pace
            status = this->process_one_message();
        }

        if (status == channel::Status::success)
        {
            backoff = 1;
        }
        else if (status == channel::Status::timeout)
        {
            // If there are no pending updates, sleep
            if (backoff < 1024)
            {
                backoff = (backoff << 1);
            }
            boost::this_fiber::sleep_for(std::chrono::microseconds(backoff));
        }
        else if (status == channel::Status::closed)
        {
            // Drop all downstream connections
            // TODO(MDD): Release all connections
        }
        else
        {
            // Should not happen
            throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Unexpected channel status in manifold: " << status));
        }
    }
}

ManifoldBase::ManifoldBase(runnable::IRunnableResources& resources,
                           std::string port_name,
                           std::unique_ptr<ManifoldTaggerBase2> tagger) :
  runnable::RunnableResourcesProvider(resources),
  m_port_name(std::move(port_name)),
  m_router_node(std::move(tagger))
{}

const PortName& ManifoldBase::port_name() const
{
    return m_port_name;
};

void ManifoldBase::start()
{
    runnable::LaunchOptions launch_options;
    launch_options.engine_factory_name = "main";
    launch_options.pe_count            = 1;
    launch_options.engines_per_pe      = 1;  // TODO(MDD): Restore to 8 after testing

    m_router_runner =
        this->runnable().launch_control().prepare_launcher(launch_options, std::move(m_router_node))->ignition();
}

void ManifoldBase::join()
{
    CHECK(m_router_runner) << "Must call start() before join()";

    m_router_runner->await_join();
}

const std::string& ManifoldBase::info() const
{
    return m_info;
}

void ManifoldBase::add_local_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
{
    throw exceptions::MrcRuntimeError("Not implemented (add_local_input)");

    // // Need to cast away the const-ness to make an edge
    // auto& tagger = const_cast<ManifoldTaggerBase&>(m_tagger_runner->runnable_as<ManifoldTaggerBase>());

    // tagger.add_input(address, input_source);
}

void ManifoldBase::add_local_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    throw exceptions::MrcRuntimeError("Not implemented (add_local_output)");

    // // First, see if we have the local port
    // if (!m_local_port_queues.contains(address))
    // {
    //     // Get the local port queue from the data plane
    //     // m_local_port_queues[address] = this->rutime().data_plane().get_local_port_queue(address);
    // }

    // // Now make an edge between the local output and the data plane queue
    // // This will use a converter to convert from a Descriptor into the ingress type
    // mrc::make_edge(m_local_port_queues[address], *output_sink);

    // // // Need to cast away the const-ness to make an edge
    // // auto& untagger = const_cast<ManifoldUnTaggerBase&>(m_untagger_runner->runnable_as<ManifoldUnTaggerBase>());

    // // untagger.add_output(address, output_sink);
}

edge::IWritableProviderBase& ManifoldBase::get_input_sink() const
{
    // Ugly, wish there was a better way
    return const_cast<edge::IWritableProviderBase&>(m_router_runner->runnable_as<edge::IWritableProviderBase>());
}

void ManifoldBase::update_policy(ManifoldPolicy&& policy)
{
    // // Need to cast away the const-ness to make an edge
    // auto& tagger   = const_cast<ManifoldTaggerBase&>(m_tagger_runner->runnable_as<ManifoldTaggerBase>());
    // auto& untagger = const_cast<ManifoldUnTaggerBase&>(m_untagger_runner->runnable_as<ManifoldUnTaggerBase>());

    // // Now update the nodes
    // tagger.update_policy(policy);

    const_cast<ManifoldTaggerBase2&>(m_router_runner->runnable_as<ManifoldTaggerBase2>())
        .update_policy(std::move(policy));
}

void ManifoldBase::update_inputs()
{
    // TODO(MDD): Delete this function
}

void ManifoldBase::update_outputs(){
    // TODO(MDD): Delete this function
};

}  // namespace mrc::manifold
