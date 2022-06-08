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

#include "../internal/architect/network_comms_manager.hpp"
#include "../internal/architect/network_events_manager.hpp"

#include <srf/channel/status.hpp>
#include <srf/codable/encode.hpp>
#include <srf/codable/encoded_object.hpp>
#include <srf/codable/fundamental_types.hpp>  // IWYU pragma: keep
#include <srf/core/bitmap.hpp>
#include <srf/core/fiber_pool.hpp>
#include <srf/core/resources.hpp>
#include <srf/memory/block.hpp>
#include <srf/memory/resources/device/cuda_malloc_resource.hpp>
#include <srf/memory/resources/host/pinned_memory_resource.hpp>
#include <srf/node/edge_builder.hpp>
#include <srf/node/operators/router.hpp>
#include <srf/node/rx_sink.hpp>
#include <srf/node/source_channel.hpp>
#include <srf/options/fiber_pool.hpp>
#include <srf/options/services.hpp>
#include <srf/options/topology.hpp>
#include <srf/runnable/engine.hpp>
#include <srf/runnable/engine_factory.hpp>
#include <srf/runnable/launch_control.hpp>
#include <srf/runnable/launch_control_config.hpp>
#include <srf/runnable/launch_options.hpp>
#include <srf/runnable/launcher.hpp>
#include <srf/runnable/runner.hpp>
#include <srf/utils/thread_local_shared_pointer.hpp>
#include "internal/system/topology.hpp"
#include "internal/ucx/context.hpp"
#include "internal/ucx/worker.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/fiber/barrier.hpp>
#include <boost/fiber/future/future.hpp>
#include <cuda/memory_resource>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>

#include <atomic>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <string>
#include <utility>

using namespace srf;

class EncodedObjectMemoryResources : public core::Resources
{
  public:
    EncodedObjectMemoryResources() :
      m_host_view(std::make_shared<memory::pinned_memory_resource>()),
      m_device_view(std::make_shared<memory::cuda_malloc_resource>(0))
    {}
    ~EncodedObjectMemoryResources() override = default;

    host_view_t host_resource_view() override
    {
        return m_host_view;
    }
    device_view_t device_resource_view() override
    {
        return m_device_view;
    }

  private:
    host_view_t m_host_view;
    device_view_t m_device_view;
};

class NetworkTests : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_topology_options.user_cpuset("0-3");
        m_topology       = Topology::Create(m_topology_options);
        m_fiber_pool_mgr = FiberPoolManager::Create(m_topology);
        m_context        = std::make_shared<ucx::Context>();
        m_mutable_nem    = std::make_unique<data_plane::Server>(m_context);

        auto memory_resources =
            std::dynamic_pointer_cast<core::Resources>(std::make_shared<EncodedObjectMemoryResources>());
        CHECK(memory_resources);

        // set up the main thread too
        utils::ThreadLocalSharedPointer<core::Resources>::set(memory_resources);

        runnable::LaunchControlConfig config;

        auto default_pool = m_fiber_pool_mgr->make_pool(CpuSet("0-2"));
        auto mgmt_pool    = m_fiber_pool_mgr->make_pool(CpuSet("3"));

        default_pool->set_thread_local_resource(memory_resources);
        mgmt_pool->set_thread_local_resource(memory_resources);

        auto default_resources = std::make_shared<runnable::FiberGroup>(default_pool);
        auto mgmt_resources    = std::make_shared<runnable::FiberGroup>(mgmt_pool);

        config.resource_groups[default_engine_factory_name()] = default_resources;
        config.resource_groups["mgmt"]                        = mgmt_resources;

        runnable::LaunchOptions default_service_options;
        default_service_options.resource_group = "mgmt";
        config.services.set_default_options(default_service_options);

        m_launch_control = std::make_shared<runnable::LaunchControl>(std::move(config));
    }

    void TearDown() override {}

    FiberPoolOptions m_fiber_pool_options;
    TopologyOptions m_topology_options;
    std::shared_ptr<Topology> m_topology;
    std::shared_ptr<FiberPoolManager> m_fiber_pool_mgr;
    std::shared_ptr<ucx::Context> m_context;
    std::shared_ptr<FiberPool> m_pool;
    std::shared_ptr<runnable::FiberEngines> m_launcher;
    std::unique_ptr<data_plane::Server> m_mutable_nem;
    std::shared_ptr<const data_plane::Server> m_nem;

    std::shared_ptr<runnable::LaunchControl> m_launch_control;
};

TEST_F(NetworkTests, NetworkEventsManagerLifeCycle)
{
    auto launcher = m_launch_control->prepare_launcher(std::move(m_mutable_nem));

    // auto& service = m_launch_control->service(runnable::SrfService::data_plane::Server);
    // service.stop();
    // service.await_join();
}

TEST_F(NetworkTests, data_plane::Server)
{
    GTEST_SKIP() << "blocked by #121";

    std::atomic<std::size_t> counter_0 = 0;
    std::atomic<std::size_t> counter_1 = 0;

    boost::fibers::barrier barrier(2);

    // create a deserialize sink which access a memory::block
    auto sink_0 = std::make_unique<node::RxSink<memory::block>>();
    sink_0->set_observer([&counter_0, &barrier](memory::block block) {
        std::free(block.data());
        ++counter_0;
        barrier.wait();
    });

    auto sink_1 = std::make_unique<node::RxSink<memory::block>>();
    sink_1->set_observer([&counter_1, &barrier](memory::block block) {
        std::free(block.data());
        ++counter_1;
        barrier.wait();
    });

    node::make_edge(m_mutable_nem->deserialize_source().source(0), *sink_0);
    node::make_edge(m_mutable_nem->deserialize_source().source(1), *sink_1);

    auto nem_worker_address = m_mutable_nem->worker_address();

    auto runner_0 = m_launch_control->prepare_launcher(std::move(sink_0))->ignition();
    auto runner_1 = m_launch_control->prepare_launcher(std::move(sink_1))->ignition();
    auto service  = m_launch_control->prepare_launcher(std::move(m_mutable_nem))->ignition();

    // NEM is running along with two Sinks attached to the deserialization router
    auto comm = std::make_unique<data_plane::Client>(std::make_shared<ucx::Worker>(m_context), *m_launch_control);
    comm->register_instance(0, nem_worker_address);

    double val = 3.14;
    codable::EncodedObject encoded_val;
    codable::encode(val, encoded_val);

    // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 0
    comm->await_send(0, 0, encoded_val);
    barrier.wait();
    EXPECT_EQ(counter_0, 1);
    EXPECT_EQ(counter_1, 0);

    // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 1
    comm->await_send(0, 1, encoded_val);
    barrier.wait();
    EXPECT_EQ(counter_0, 1);
    EXPECT_EQ(counter_1, 1);

    service->stop();
    service->await_join();

    runner_0->await_join();
    runner_1->await_join();

    comm.reset();
}
