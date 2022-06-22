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

#include "internal/architect/executor_architect.hpp"

#include "internal/control_plane/server.hpp"
#include "internal/executor.hpp"

#include <srf/forward.hpp>
#include <srf/options/options.hpp>
#include <srf/options/topology.hpp>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <ostream>

using namespace srf;
using namespace internal;

class TestArchitect : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        DVLOG(10) << "test harness: construct architect server";
        m_server = std::make_unique<control_plane::Server>(13337);
    }

    void TearDown() override
    {
        DVLOG(10) << "test harness: shutdown architect server";
        m_server->shutdown();
    }

    static std::shared_ptr<Options> make_options(std::function<void(Options&)> updater = nullptr)
    {
        auto options = std::make_shared<Options>();
        options->architect_url("127.0.0.1:13337");
        options->topology().user_cpuset("0-3");

        if (updater)
        {
            updater(*options);
        }

        return options;
    }

    static std::unique_ptr<ArchitectExecutor> make_executor(std::shared_ptr<Options> options = nullptr)
    {
        if (options == nullptr)
        {
            options = make_options();
        }

        return std::make_unique<ArchitectExecutor>(options);
    }

    std::unique_ptr<control_plane::Server> m_server;
};

TEST_F(TestArchitect, LifeCycle)
{
    auto executor = make_executor();
    executor->start();
    executor->stop();
    executor->join();
}

TEST_F(TestArchitect, LifeCycleMultiNode)
{
    auto rank_0 = make_executor(make_options([](Options& opts) { opts.topology().user_cpuset("0-2"); }));
    auto rank_1 = make_executor(make_options([](Options& opts) { opts.topology().user_cpuset("3-4"); }));
    rank_0->start();
    rank_1->start();
    rank_0->stop();
    rank_1->stop();
    rank_0->join();
    rank_1->join();
}

TEST_F(TestArchitect, UCXMultiNodeEvents)
{
    auto rank_0 = make_executor(make_options([](Options& opts) { opts.topology().user_cpuset("0-2"); }));
    auto rank_1 = make_executor(make_options([](Options& opts) { opts.topology().user_cpuset("3-4"); }));
    rank_0->start();
    rank_1->start();

    //

    // shutdown
    rank_0->stop();
    rank_1->stop();
    rank_0->join();
    rank_1->join();
}
