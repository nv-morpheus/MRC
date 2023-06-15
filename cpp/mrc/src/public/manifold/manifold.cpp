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

#include "mrc/edge/edge_builder.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>

#include <ostream>
#include <string>
#include <utility>

namespace mrc::manifold {

Manifold::Manifold(PortName port_name, runnable::IRunnableResources& resources) :
  m_port_name(std::move(port_name)),
  m_resources(resources)

{}

Manifold::~Manifold() = default;

const PortName& Manifold::port_name() const
{
    return m_port_name;
}

runnable::IRunnableResources& Manifold::resources()
{
    return m_resources;
}

const std::string& Manifold::info() const
{
    return m_info;
}

void Manifold::add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to upstream segment " << segment::info(address);
    do_add_input(address, input_source);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to upstream segment "
              << segment::info(address);
}

void Manifold::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    DVLOG(3) << "manifold " << this->port_name() << ": connecting to downstream segment " << segment::info(address);
    do_add_output(address, output_sink);
    DVLOG(10) << "manifold " << this->port_name() << ": completed connection to downstream segment "
              << segment::info(address);
}

void ManifoldNodeBase::add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
{
    boost::fibers::packaged_task<void()> update_task([this, address, input_source] {
        mrc::make_edge(*input_source, *this);
    });

    auto update_future = update_task.get_future();

    CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

    // Before continuing, wait for the update to be processed
    update_future.get();
}

void ManifoldNodeBase::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    boost::fibers::packaged_task<void()> update_task([this, address, output_sink] {
        mrc::make_edge(this->get_output(address), *output_sink);
    });

    auto update_future = update_task.get_future();

    CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

    // Before continuing, wait for the update to be processed
    update_future.get();
}

void ManifoldNodeBase::run(runnable::Context& ctx)
{
    std::uint64_t backoff = 128;

    while (m_is_running)
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

        // Process one element. This will pull one off the queue, process it, and return the status
        auto status = this->process_one();

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
        else
        {
            // Should not happen
            throw exceptions::MrcRuntimeError(MRC_CONCAT_STR("Unexpected channel status in manifold: " << status));
        }
    }
}

void ManifoldTaggerBase::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    // Call the base and then update the list of available tags
    ManifoldNodeBase::add_output(address, output_sink);

    m_available_outputs.push_back(address);

    // Now make sure we dont have any duplicates
    std::sort(m_available_outputs.begin(), m_available_outputs.end());
    m_available_outputs.erase(std::unique(m_available_outputs.begin(), m_available_outputs.end()),
                              m_available_outputs.end());
}

SegmentAddress ManifoldTaggerBase::get_next_tag()
{
    // Just loop over all options for now
    return m_available_outputs[m_msg_counter++ % m_available_outputs.size()];
}

ManifoldBase::ManifoldBase(runnable::IRunnableResources& resources,
                           std::string port_name,
                           std::unique_ptr<ManifoldTaggerBase> tagger,
                           std::unique_ptr<ManifoldUnTaggerBase> untagger) :
  runnable::RunnableResourcesProvider(resources),
  m_port_name(std::move(port_name)),
  m_tagger_node(std::move(tagger)),
  m_untagger_node(std::move(untagger))
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

    m_tagger_runner =
        this->runnable().launch_control().prepare_launcher(launch_options, std::move(m_tagger_node))->ignition();
    m_untagger_runner =
        this->runnable().launch_control().prepare_launcher(launch_options, std::move(m_untagger_node))->ignition();
}

void ManifoldBase::join()
{
    CHECK(m_tagger_runner) << "Must call start() before join()";
    CHECK(m_untagger_runner) << "Must call start() before join()";

    m_tagger_runner->await_join();
    m_untagger_runner->await_join();
}

const std::string& ManifoldBase::info() const
{
    return m_info;
}

void ManifoldBase::add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
{
    // Need to cast away the const-ness to make an edge
    auto& tagger = const_cast<ManifoldTaggerBase&>(m_tagger_runner->runnable_as<ManifoldTaggerBase>());

    tagger.add_input(address, input_source);
}

void ManifoldBase::add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
{
    // Need to cast away the const-ness to make an edge
    auto& tagger   = const_cast<ManifoldTaggerBase&>(m_tagger_runner->runnable_as<ManifoldTaggerBase>());
    auto& untagger = const_cast<ManifoldUnTaggerBase&>(m_untagger_runner->runnable_as<ManifoldUnTaggerBase>());

    // Need to add a local connection to the tagger first
    tagger.add_output(address, &untagger);

    untagger.add_output(address, output_sink);
}

void ManifoldBase::update_inputs()
{
    // TODO(MDD): Delete this function
}

void ManifoldBase::update_outputs(){
    // TODO(MDD): Delete this function
};

}  // namespace mrc::manifold
