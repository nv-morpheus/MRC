/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "./node/test_nodes.hpp"
#include "./test_mrc.hpp"

#include "mrc/channel/buffered_channel.hpp"  // IWYU pragma: keep
#include "mrc/edge/edge.hpp"                 // for Edge
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_holder.hpp"  // for EdgeHolder
#include "mrc/edge/edge_readable.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/operators/round_robin_router_typeless.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/source_properties.hpp"

#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>  // for observable_member

#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

// IWYU pragma: no_forward_declare mrc::channel::Channel

// IWYU thinks we need vector for make_segment
// IWYU pragma: no_include <vector>

using namespace std::chrono_literals;

TEST_CLASS(Edges);

using TestEdgesDeathTest = TestEdges;  // NOLINT(readability-identifier-naming)

namespace mrc {

TEST_F(TestEdgesDeathTest, NodeDestroyedBeforeEdge)
{
    // Reset the sink before the source which will cause an exception
    EXPECT_DEATH(
        {
            auto source = std::make_shared<node::TestSource<int>>();
            auto sink   = std::make_shared<node::TestSink<int>>();

            mrc::make_edge(*source, *sink);
            sink.reset();
        },
        "");
}

TEST_F(TestEdges, SourceToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToSinkUpcast)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<float>>();

    mrc::make_edge(*source, *sink);

    source->run();
    sink->run();

    std::vector<float> source_float_vals;

    for (const auto& v : source->get_init_values())
    {
        source_float_vals.push_back(v);
    }

    EXPECT_EQ(source_float_vals, sink->get_values());
}

TEST_F(TestEdges, SourceToSinkTypeless)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge_typeless(*source, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    source->run();
    node->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToNodeToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node1  = std::make_shared<node::TestNode<int>>();
    auto node2  = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *node1);
    mrc::make_edge(*node1, *node2);
    mrc::make_edge(*node2, *sink);

    source->run();
    node1->run();
    node2->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToSinkMultiFail)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *sink1);
    EXPECT_THROW(mrc::make_edge(*source, *sink2), std::runtime_error);

    source.reset();
    sink1.reset();
    sink2.reset();
}

TEST_F(TestEdges, SourceToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *sink);

    source->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceComponentToSink)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *sink);

    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceComponentToNodeToSink)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    node->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToNodeComponentToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::NodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    source->run();
    node->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToNodeComponentToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    source->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToRxNodeComponentToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestRxNodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    node->make_stream([=](rxcpp::observable<int> input) {
        return input.map([](int i) {
            return i * 2;
        });
    });

    source->run();

    EXPECT_TRUE(node->stream_fn_called);

    EXPECT_EQ((std::vector<int>{0, 2, 4}), sink->get_values());
}

TEST_F(TestEdges, SourceComponentToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    node->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToQueueToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *queue);
    mrc::make_edge(*queue, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToQueueToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *queue);
    mrc::make_edge(*queue, *node);
    mrc::make_edge(*node, *sink);

    source->run();
    node->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToQueueToMultiSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *queue);
    mrc::make_edge(*queue, *sink1);
    mrc::make_edge(*queue, *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ(source->get_init_values(), sink1->get_values());
    EXPECT_EQ(std::vector<int>{}, sink2->get_values());
}

TEST_F(TestEdges, SourceToQueueToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *queue);
    mrc::make_edge(*queue, *sink1);
    mrc::make_edge(*queue, *node);
    mrc::make_edge(*node, *sink2);

    source->run();
    node->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{}), sink1->get_values());
    EXPECT_EQ(source->get_init_values(), sink2->get_values());
}

TEST_F(TestEdges, SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestRouter>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestEdges, SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestRouter>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("odd"), *sink1);
    mrc::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{1}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{0, 2}), sink2->get_values());
}

TEST_F(TestEdges, SourceToRoundRobinRouterTypelessToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::RoundRobinRouterTypeless>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router, *sink1);
    mrc::make_edge(*router, *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ((std::vector<int>{0, 2}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1}), sink2->get_values());
}

TEST_F(TestEdges, SourceToDynamicRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>(10);
    auto router = std::make_shared<node::TestDynamicRouter<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();
    auto sink3  = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *router);
    mrc::make_edge(*router->get_source("1"), *sink1);
    mrc::make_edge(*router->get_source("2"), *sink2);
    mrc::make_edge(*router->get_source("3"), *sink3);

    source->run();
    sink1->run();
    sink2->run();
    sink3->run();

    EXPECT_EQ((std::vector<int>{0, 3, 6, 9}), sink1->get_values());
    EXPECT_EQ((std::vector<int>{1, 4, 7}), sink2->get_values());
    EXPECT_EQ((std::vector<int>{2, 5, 8}), sink3->get_values());
}

TEST_F(TestEdges, SourceToBroadcastToSink)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToBroadcastTypelessToSinkSinkFirst)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::BroadcastTypeless>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*broadcast, *sink);
    mrc::make_edge(*source, *broadcast);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToBroadcastTypelessToSinkSourceFirst)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::BroadcastTypeless>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToBroadcastTypelessToDifferentSinks)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::BroadcastTypeless>();
    auto sink1     = std::make_shared<node::TestSink<int>>();
    auto sink2     = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink1);
    mrc::make_edge(*broadcast, *sink2);

    source->run();
}

TEST_F(TestEdges, SourceToMultipleBroadcastTypelessToSinkSinkFirst)
{
    auto source     = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink       = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*broadcast2, *sink);
    mrc::make_edge(*broadcast1, *broadcast2);
    mrc::make_edge(*source, *broadcast1);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, SourceToMultipleBroadcastTypelessToSinkSourceFirst)
{
    auto source     = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink       = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *broadcast1);
    mrc::make_edge(*broadcast1, *broadcast2);
    mrc::make_edge(*broadcast2, *sink);

    source->run();
    sink->run();

    EXPECT_EQ(source->get_init_values(), sink->get_values());
}

TEST_F(TestEdges, MultiSourceToMultipleBroadcastTypelessToMultiSink)
{
    auto source1    = std::make_shared<node::TestSource<int>>();
    auto source2    = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink1      = std::make_shared<node::TestSink<int>>();
    auto sink2      = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source1, *broadcast1);
    mrc::make_edge(*source2, *broadcast1);
    mrc::make_edge(*broadcast1, *broadcast2);
    mrc::make_edge(*broadcast2, *sink1);
    mrc::make_edge(*broadcast2, *sink2);

    source1->run();
    source2->run();
    sink1->run();
    sink2->run();

    auto expected = source1->get_init_values();
    expected.insert(expected.end(), source2->get_init_values().begin(), source2->get_init_values().end());

    EXPECT_EQ(expected, sink1->get_values());
    EXPECT_EQ(expected, sink2->get_values());
}

TEST_F(TestEdges, SourceToBroadcastToMultiSink)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSink<int>>();
    auto sink2     = std::make_shared<node::TestSink<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink1);
    mrc::make_edge(*broadcast, *sink2);

    source->run();
    sink1->run();
    sink2->run();

    EXPECT_EQ(source->get_init_values(), sink1->get_values());
    EXPECT_EQ(source->get_init_values(), sink2->get_values());
}

TEST_F(TestEdges, SourceToBroadcastToDifferentSinks)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSink<int>>();
    auto sink2     = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink1);
    mrc::make_edge(*broadcast, *sink2);

    source->run();
    sink1->run();

    EXPECT_EQ(source->get_init_values(), sink1->get_values());
    EXPECT_EQ(source->get_init_values(), sink2->get_values());
}

TEST_F(TestEdges, SourceToBroadcastToSinkComponents)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSinkComponent<int>>();
    auto sink2     = std::make_shared<node::TestSinkComponent<int>>();

    mrc::make_edge(*source, *broadcast);
    mrc::make_edge(*broadcast, *sink1);
    mrc::make_edge(*broadcast, *sink2);

    source->run();

    EXPECT_EQ(source->get_init_values(), sink1->get_values());
    EXPECT_EQ(source->get_init_values(), sink2->get_values());
}

TEST_F(TestEdges, SourceComponentDoubleToSinkFloat)
{
    auto source = std::make_shared<node::TestSourceComponent<double>>();
    auto sink   = std::make_shared<node::TestSink<float>>();

    mrc::make_edge(*source, *sink);

    sink->run();

    EXPECT_EQ((std::vector<float>{0, 1, 2}), sink->get_values());
}

TEST_F(TestEdges, SourceToNull)
{
    auto source = std::make_shared<node::TestSource<int>>();

    source->run();
}

TEST_F(TestEdges, SourceToNodeToNull)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();

    mrc::make_edge(*source, *node);

    source->run();
    node->run();
}

TEST_F(TestEdges, CreateAndDestroy)
{
    {
        auto x = std::make_shared<node::TestSource<int>>();
    }

    {
        auto x = std::make_shared<node::TestNode<int>>();
    }

    {
        auto x = std::make_shared<node::TestSink<int>>();
    }

    {
        auto x = std::make_shared<node::TestSourceComponent<int>>();
    }

    {
        auto x = std::make_shared<node::TestNodeComponent<int>>();
    }

    {
        auto x = std::make_shared<node::TestRxNodeComponent<int>>();
    }

    {
        auto x = std::make_shared<node::TestSinkComponent<int>>();
    }

    {
        auto x = std::make_shared<node::Broadcast<int>>();
    }
}

TEST_F(TestEdges, EdgeTapWAcceptorWProvider)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    // Original edge
    mrc::make_edge(*source, *sink);

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source, *sink, *node, *node);

    source->run();
    node->run();
    sink->run();
}

TEST_F(TestEdges, EdgeTapRProviderRAcceptor)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto source_rp = std::dynamic_pointer_cast<edge::IReadableProvider<int>>(source);

    auto node = std::make_shared<node::TestNode<int>>();

    auto sink    = std::make_shared<node::TestSink<int>>();
    auto sink_ra = std::dynamic_pointer_cast<edge::IReadableAcceptor<int>>(sink);

    // Original edge
    mrc::make_edge(*source_rp, *sink_ra);

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source_rp, *sink_ra, *node, *node);

    source->run();
    node->run();
    sink->run();
}

TEST_F(TestEdges, EdgeTapWithComponentSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    // Original edge
    mrc::make_edge(*source, *sink);

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source, *sink, *node, *node);

    source->run();
    node->run();
}

TEST_F(TestEdges, EdgeTapWithSourceComponent)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    // Original edge
    mrc::make_edge(*source, *sink);

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source, *sink, *node, *node);

    node->run();
    sink->run();
}

TEST_F(TestEdges, EdgeTapWithSpliceComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    // Original edge
    mrc::make_edge(*source, *sink);

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source, *sink, *node, *node);

    source->run();
    sink->run();
}

TEST_F(TestEdges, EdgeTapWithSpliceRxComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestRxNodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    // Original edge
    mrc::make_edge(*source, *sink);

    node->make_stream([=](rxcpp::observable<int> input) {
        return input.map([](int i) {
            return i * 2;
        });
    });

    // Tap edge
    mrc::edge::EdgeBuilder::splice_edge<int>(*source, *sink, *node, *node);

    source->run();
    sink->run();

    EXPECT_TRUE(node->stream_fn_called);
}

template <typename T>
class TestEdgeHolder : public edge::EdgeHolder<T>
{
  public:
    bool has_active_connection() const
    {
        return this->check_active_connection(false);
    }

    void call_release_edge_connection()
    {
        this->release_edge_connection();
    }

    void call_init_owned_edge(std::shared_ptr<edge::Edge<T>> edge)
    {
        this->init_owned_edge(std::move(edge));
    }
};

TEST_F(TestEdges, EdgeHolderIsConnected)
{
    TestEdgeHolder<int> edge_holder;
    auto edge = std::make_shared<edge::Edge<int>>();
    EXPECT_FALSE(edge_holder.has_active_connection());

    edge_holder.call_init_owned_edge(edge);
    EXPECT_FALSE(edge_holder.has_active_connection());

    edge_holder.call_release_edge_connection();
    EXPECT_FALSE(edge_holder.has_active_connection());
}
}  // namespace mrc
