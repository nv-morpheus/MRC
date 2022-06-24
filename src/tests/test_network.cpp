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

#include "internal/resources/manager.hpp"
#include "internal/system/system.hpp"
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

static std::shared_ptr<internal::system::System> make_system(std::function<void(Options&)> updater = nullptr)
{
    auto options = std::make_shared<Options>();
    if (updater)
    {
        updater(*options);
    }

    return internal::system::make_system(std::move(options));
}

class TestNetwork : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_resources = std::make_unique<internal::resources::Manager>(internal::system::SystemProvider(
            make_system([](Options& options) { options.architect_url("localhost:13337"); })));

        // options.topology().user_cpuset("0-3");
        // options.topology().restrict_gpus(true);
        // options.engine_factories().set_engine_factory_options("thread_pool", [](EngineFactoryOptions& options) {
        // options.engine_type   = runnable::EngineType::Thread;
        // options.allow_overlap = false;
        // options.cpu_count     = 2;
    }

    void TearDown() override
    {
        m_resources.reset();
    }

    std::unique_ptr<internal::resources::Manager> m_resources;
};

TEST_F(TestNetwork, LifeCycle) {}

// TEST_F(TestNetwork, NetworkEventsManagerLifeCycle)
// {
//     auto launcher = m_launch_control->prepare_launcher(std::move(m_mutable_nem));

//     // auto& service = m_launch_control->service(runnable::SrfService::data_plane::Server);
//     // service.stop();
//     // service.await_join();
// }

// TEST_F(TestNetwork, data_plane::Server)
// {
//     GTEST_SKIP() << "blocked by #121";

//     std::atomic<std::size_t> counter_0 = 0;
//     std::atomic<std::size_t> counter_1 = 0;

//     boost::fibers::barrier barrier(2);

//     // create a deserialize sink which access a memory::block
//     auto sink_0 = std::make_unique<node::RxSink<memory::block>>();
//     sink_0->set_observer([&counter_0, &barrier](memory::block block) {
//         std::free(block.data());
//         ++counter_0;
//         barrier.wait();
//     });

//     auto sink_1 = std::make_unique<node::RxSink<memory::block>>();
//     sink_1->set_observer([&counter_1, &barrier](memory::block block) {
//         std::free(block.data());
//         ++counter_1;
//         barrier.wait();
//     });

//     node::make_edge(m_mutable_nem->deserialize_source().source(0), *sink_0);
//     node::make_edge(m_mutable_nem->deserialize_source().source(1), *sink_1);

//     auto nem_worker_address = m_mutable_nem->worker_address();

//     auto runner_0 = m_launch_control->prepare_launcher(std::move(sink_0))->ignition();
//     auto runner_1 = m_launch_control->prepare_launcher(std::move(sink_1))->ignition();
//     auto service  = m_launch_control->prepare_launcher(std::move(m_mutable_nem))->ignition();

//     // NEM is running along with two Sinks attached to the deserialization router
//     auto comm = std::make_unique<data_plane::Client>(std::make_shared<ucx::Worker>(m_context),
//     *m_launch_control); comm->register_instance(0, nem_worker_address);

//     double val = 3.14;
//     codable::EncodedObject encoded_val;
//     codable::encode(val, encoded_val);

//     // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 0
//     comm->await_send(0, 0, encoded_val);
//     barrier.wait();
//     EXPECT_EQ(counter_0, 1);
//     EXPECT_EQ(counter_1, 0);

//     // send a ucx tagged message with a memory block of a fixed sized with a destination port address of 1
//     comm->await_send(0, 1, encoded_val);
//     barrier.wait();
//     EXPECT_EQ(counter_0, 1);
//     EXPECT_EQ(counter_1, 1);

//     service->stop();
//     service->await_join();

//     runner_0->await_join();
//     runner_1->await_join();

//     comm.reset();
// }
