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

#include "mrc/core/utils.hpp"
#include "mrc/exceptions/runtime_error.hpp"
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

// SegmentAddress ManifoldTaggerBase::get_next_tag()
// {
//     return this->current_policy().get_next_tag();
// }

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
    // Get exclusive access to the lock to prevent modification
    std::unique_lock lock(m_output_mutex);

    auto next_keys = extract_keys(policy.outputs);

    auto [to_add, to_remove] = compare_difference(this->writable_edge_keys(),
                                                  std::vector<InstanceID>{next_keys.begin(), next_keys.end()});

    for (const auto& key : to_remove)
    {
        this->release_writable_edge(key);
    }

    for (const auto& key : to_add)
    {
        auto& output = policy.outputs.at(key);

        this->add_output(key, output.is_local, output.edge);
    }

    m_current_policy = std::move(policy);
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
    // runnable::LaunchOptions launch_options;
    // launch_options.engine_factory_name = "main";
    // launch_options.pe_count            = 1;
    // launch_options.engines_per_pe      = 1;  // TODO(MDD): Restore to 8 after testing

    // m_tagger_runner =
    //     this->runnable().launch_control().prepare_launcher(launch_options, std::move(m_tagger_node))->ignition();
    // m_untagger_runner =
    //     this->runnable().launch_control().prepare_launcher(launch_options, std::move(m_untagger_node))->ignition();
}

void ManifoldBase::join()
{
    // CHECK(m_tagger_runner) << "Must call start() before join()";
    // CHECK(m_untagger_runner) << "Must call start() before join()";

    // m_tagger_runner->await_join();
    // m_untagger_runner->await_join();
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
    return *m_router_node;
}

void ManifoldBase::update_policy(ManifoldPolicy&& policy)
{
    // // Need to cast away the const-ness to make an edge
    // auto& tagger   = const_cast<ManifoldTaggerBase&>(m_tagger_runner->runnable_as<ManifoldTaggerBase>());
    // auto& untagger = const_cast<ManifoldUnTaggerBase&>(m_untagger_runner->runnable_as<ManifoldUnTaggerBase>());

    // // Now update the nodes
    // tagger.update_policy(policy);

    m_router_node->update_policy(std::move(policy));
}

void ManifoldBase::update_inputs()
{
    // TODO(MDD): Delete this function
}

void ManifoldBase::update_outputs(){
    // TODO(MDD): Delete this function
};

}  // namespace mrc::manifold
