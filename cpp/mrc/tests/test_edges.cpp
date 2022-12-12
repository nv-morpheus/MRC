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

#include "test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/core/addresses.hpp"
#include "mrc/core/executor.hpp"
#include "mrc/node/channel_holder.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/edge_channel.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/combine_latest.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/rx_subscribable.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/placement.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/pipeline.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/types.hpp"

#include <glog/logging.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <gtest/internal/gtest-internal.h>
#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

// IWYU thinks we need vector for make_segment
// IWYU pragma: no_include <vector>

using namespace std::chrono_literals;

TEST_CLASS(Edges);

namespace mrc::node {

template <typename T>
class EdgeReadableLambda : public IEdgeReadable<T>
{
  public:
    EdgeReadableLambda(std::function<channel::Status(T&)>&& on_await_read,
                       std::function<void()>&& on_complete = nullptr) :
      m_on_await_read(std::move(on_await_read)),
      m_on_complete(std::move(on_complete))
    {}

    ~EdgeReadableLambda() override
    {
        if (m_on_complete)
        {
            m_on_complete();
        }
    }

    channel::Status await_read(T& t) override
    {
        return m_on_await_read(t);
    }

  private:
    std::function<channel::Status(T&)> m_on_await_read;
    std::function<void()> m_on_complete;
};

template <typename T>
class EdgeWritableLambda : public IEdgeWritable<T>
{
  public:
    EdgeWritableLambda(std::function<channel::Status(T&&)>&& on_await_write,
                       std::function<void()>&& on_complete = nullptr) :
      m_on_await_write(std::move(on_await_write)),
      m_on_complete(std::move(on_complete))
    {}

    ~EdgeWritableLambda() override
    {
        if (m_on_complete)
        {
            m_on_complete();
        }
    }

    channel::Status await_write(T&& t) override
    {
        return m_on_await_write(std::move(t));
    }

  private:
    std::function<channel::Status(T&&)> m_on_await_write;
    std::function<void()> m_on_complete;
};

template <typename T>
class TestSource : public IngressAcceptor<T>, public EgressProvider<T>, public SourceChannel<T>
{
  public:
    TestSource()
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void run()
    {
        auto output = this->get_writable_edge();

        for (int i = 0; i < 3; i++)
        {
            if (output->await_write(T(i)) != channel::Status::success)
            {
                break;
            }
        }

        this->release_edge_connection();
    }
};

template <typename T>
class TestNode : public IngressProvider<T>,
                 public EgressAcceptor<T>,
                 public IngressAcceptor<T>,
                 public EgressProvider<T>,
                 public SinkChannel<T>,
                 public SourceChannel<T>
{
  public:
    TestNode()
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        EdgeChannel<T> edge_channel(std::move(channel));

        SinkChannel<T>::do_set_channel(edge_channel);
        SourceChannel<T>::do_set_channel(edge_channel);
    }

    void run()
    {
        auto input  = this->get_readable_edge();
        auto output = this->get_writable_edge();

        int t;

        while (input->await_read(t) == channel::Status::success)
        {
            VLOG(10) << "Node got value: " << t;

            output->await_write(std::move(t));
        }

        VLOG(10) << "Node exited run";

        SinkChannel<T>::release_edge_connection();
        SourceChannel<T>::release_edge_connection();
    }
};

template <typename T>
class TestSink : public IngressProvider<T>, public EgressAcceptor<T>, public SinkChannel<T>
{
  public:
    TestSink()
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void run()
    {
        auto input = this->get_readable_edge();

        T t;

        while (input->await_read(t) == channel::Status::success)
        {
            VLOG(10) << "Sink got value";
        }

        VLOG(10) << "Sink exited run";

        this->release_edge_connection();
    }
};

template <typename T>
class TestQueue : public IngressProvider<T>, public EgressProvider<T>
{
  public:
    TestQueue()
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        EdgeChannel<T> edge_channel(std::move(channel));

        SinkProperties<T>::init_edge(edge_channel.get_writer());
        SourceProperties<T>::init_edge(edge_channel.get_reader());
    }
};

template <typename T>
class TestSourceComponent : public EgressProvider<T>
{
  public:
    TestSourceComponent()
    {
        this->init_edge(std::make_shared<EdgeReadableLambda<T>>(
            [this](int& t) {
                // Call this object
                return this->await_read(t);
            },
            [this]() { this->on_complete(); }));
    }

    channel::Status await_read(int& t)
    {
        t = m_value++;

        VLOG(10) << "TestSourceComponent emmitted value: " << t;

        // Close after 3
        return m_value >= 3 ? channel::Status::closed : channel::Status::success;
    }

    void on_complete()
    {
        VLOG(10) << "TestSourceComponent completed";
    }

  private:
    int m_value{1};
};

template <typename T>
class TestNodeComponent : public NodeComponent<T, T>
{
  public:
    TestNodeComponent()
    {
        IngressProvider<T>::init_edge(std::make_shared<EdgeWritableLambda<T>>(
            [this](int&& t) {
                // Call this object
                return this->on_next(std::move(t));
            },
            [this]() {
                // Call complete, and then drop the downstream edge
                this->on_complete();

                // TODO(MDD): Release downstream edge
                SourceProperties<T>::release_edge_connection();
            }));
    }

    ~TestNodeComponent()
    {
        // Debug print
        VLOG(10) << "Destroying TestNodeComponent";
    }

    channel::Status on_next(int&& t)
    {
        VLOG(10) << "TestNodeComponent got value: " << t;

        return this->get_writable_edge()->await_write(t + 1);
    }

    void on_complete()
    {
        VLOG(10) << "TestSinkComponent completed";
    }
};

template <typename T>
class TestSinkComponent : public IngressProvider<T>
{
  public:
    TestSinkComponent()
    {
        this->init_edge(std::make_shared<EdgeWritableLambda<T>>(
            [this](int&& t) {
                // Call this object
                return this->await_write(std::move(t));
            },
            [this]() { this->on_complete(); }));
    }

    channel::Status await_write(int&& t)
    {
        VLOG(10) << "TestSinkComponent got value: " << t;

        return channel::Status::success;
    }

    void on_complete()
    {
        VLOG(10) << "TestSinkComponent completed";
    }
};

template <typename T>
class TestRouter : public Router<std::string, int>
{
  protected:
    virtual std::string determine_key_for_value(const int& t) override
    {
        return "test";
    }
};

// class TestRouter : public IngressProvider<int>
// {
//     class UpstreamEdge : public IEdgeWritable<int>, public MultiSourceProperties<int, std::string>
//     {
//       public:
//         UpstreamEdge(TestRouter& parent) : m_parent(parent) {}

//         ~UpstreamEdge() override
//         {
//             m_parent.on_complete();
//         }

//         channel::Status await_write(int&& t) override
//         {
//             VLOG(10) << "TestRouter got value: " << t;

//             // Determine key for value
//             auto key = m_parent.determine_key_for_value(t);

//             return this->get_writable_edge(key)->await_write(std::move(t));
//         }

//         void add_downstream(std::string key, std::shared_ptr<IEdgeWritable<int>> downstream)
//         {
//             this->set_edge(std::move(key), std::move(downstream));
//         }

//       private:
//         TestRouter& m_parent;
//     };

//     class DownstreamEdge : public IIngressAcceptor<int>
//     {
//       public:
//         DownstreamEdge(std::weak_ptr<UpstreamEdge> upstream, std::string key) :
//           m_upstream(std::move(upstream)),
//           m_key(std::move(key))
//         {}

//         void set_ingress(std::shared_ptr<IEdgeWritable<int>> ingress) override
//         {
//             // Get a lock to the upstream edge
//             if (auto upstream = m_upstream.lock())
//             {
//                 upstream->add_downstream(m_key, ingress);
//             }
//             else
//             {
//                 LOG(ERROR) << "Could not set ingress to upstream edge. Upstream edge has been destroyed.";
//             }
//         }

//       private:
//         std::weak_ptr<UpstreamEdge> m_upstream;
//         std::string m_key;
//     };

//   public:
//     TestRouter()
//     {
//         auto upstream = std::make_shared<UpstreamEdge>(*this);

//         // Save it to avoid casting
//         m_upstream = upstream;

//         IngressProvider<int>::init_edge(upstream);
//     }

//     std::shared_ptr<IIngressAcceptor<int>> get_source(const std::string& key) const
//     {
//         auto found = m_downstream.find(key);

//         if (found == m_downstream.end())
//         {
//             auto new_source = std::make_shared<DownstreamEdge>(m_upstream, key);

//             const_cast<TestRouter*>(this)->m_downstream[key] = new_source;

//             return new_source;
//         }

//         return found->second;
//     }

//     void on_complete()
//     {
//         VLOG(10) << "TestRouter completed";
//     }

//   protected:
//     std::string determine_key_for_value(const int& t)
//     {
//         return "test";
//     }

//   private:
//     std::weak_ptr<UpstreamEdge> m_upstream;
//     std::map<std::string, std::shared_ptr<DownstreamEdge>> m_downstream;
// };

template <typename T>
class TestConditional : public IngressProvider<int>, public IngressAcceptor<int>
{
  public:
    TestConditional()
    {
        IngressProvider<int>::init_edge(std::make_shared<EdgeWritableLambda<int>>(
            [this](int&& t) {
                // Call this object
                return this->on_next(std::move(t));
            },
            [this]() {
                // Call complete, and then drop the downstream edge
                this->on_complete();

                // TODO(MDD): Release downstream edge
                SourceProperties<int>::release_edge_connection();
            }));
    }

    ~TestConditional()
    {
        // Debug print
        VLOG(10) << "Destroying TestConditional";
    }

    channel::Status on_next(int&& t)
    {
        VLOG(10) << "TestConditional got value: " << t;

        // Skip on condition
        if (t % 2 == 0)
        {
            return channel::Status::success;
        }

        return this->get_writable_edge()->await_write(t + 1);
    }

    void on_complete()
    {
        VLOG(10) << "TestConditional completed";
    }
};

// class TestBroadcast : public IngressProvider<int>, public IIngressAcceptor<int>
// {
//     class BroadcastEdge : public IEdgeWritable<int>, public MultiSourceProperties<int, size_t>
//     {
//       public:
//         BroadcastEdge(TestBroadcast& parent) : m_parent(parent) {}

//         ~BroadcastEdge()
//         {
//             m_parent.on_complete();
//         }

//         channel::Status await_write(int&& t) override
//         {
//             VLOG(10) << "BroadcastEdge got value: " << t;

//             for (size_t i = this->edge_count() - 1; i > 0; i--)
//             {
//                 // Make a copy
//                 int x = t;

//                 auto response = this->get_writable_edge(i)->await_write(std::move(x));

//                 if (response != channel::Status::success)
//                 {
//                     return response;
//                 }
//             }

//             // Write index 0 last
//             return this->get_writable_edge(0)->await_write(std::move(t));
//         }

//         void add_downstream(std::shared_ptr<IEdgeWritable<int>> downstream)
//         {
//             auto edge_count = this->edge_count();

//             this->set_edge(edge_count, downstream);
//         }

//       private:
//         TestBroadcast& m_parent;
//         // std::vector<std::shared_ptr<EdgeWritable<int>>> m_outputs;
//     };

//   public:
//     TestBroadcast()
//     {
//         auto edge = std::make_shared<BroadcastEdge>(*this);

//         // Save to avoid casting
//         m_edge = edge;

//         IngressProvider<int>::init_edge(edge);
//     }

//     ~TestBroadcast()
//     {
//         // Debug print
//         VLOG(10) << "Destroying TestBroadcast";
//     }

//     void set_ingress(std::shared_ptr<IEdgeWritable<int>> ingress) override
//     {
//         if (auto e = m_edge.lock())
//         {
//             e->add_downstream(ingress);
//         }
//         else
//         {
//             LOG(ERROR) << "Edge was destroyed";
//         }
//     }

//     void set_ingress_typeless(std::shared_ptr<EdgeTag> ingress) override
//     {
//         this->set_ingress(std::dynamic_pointer_cast<IEdgeWritable<int>>(ingress));
//     }

//     void on_complete()
//     {
//         VLOG(10) << "TestBroadcast completed";
//     }

//   private:
//     std::weak_ptr<BroadcastEdge> m_edge;
// };

}  // namespace mrc::node

namespace mrc {

TEST_F(TestEdges, NodeDestroyedBeforeEdge)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *sink);

    // Reset the sink before the source which will cause an exception
    EXPECT_DEATH(sink.reset(), "");
}

TEST_F(TestEdges, SourceToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToSinkUpcast)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<float>>();

    node::make_edge(*source, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToSinkTypeless)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge_typeless(*source, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    source->run();
    node->run();
    sink->run();
}

TEST_F(TestEdges, SourceToNodeToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node1  = std::make_shared<node::TestNode<int>>();
    auto node2  = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *node1);
    node::make_edge(*node1, *node2);
    node::make_edge(*node2, *sink);

    source->run();
    node1->run();
    node2->run();
    sink->run();
}

TEST_F(TestEdges, SourceToSinkMultiFail)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *sink1);
    EXPECT_THROW(node::make_edge(*source, *sink2), std::runtime_error);
}

TEST_F(TestEdges, SourceToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *sink);

    source->run();
}

TEST_F(TestEdges, SourceComponentToSink)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *sink);

    sink->run();
}

TEST_F(TestEdges, SourceComponentToNodeToSink)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    node->run();
    sink->run();
}

TEST_F(TestEdges, SourceToNodeComponentToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::NodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    source->run();
    node->run();
}

TEST_F(TestEdges, SourceToNodeComponentToSinkComponent)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto node   = std::make_shared<node::TestNodeComponent<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    source->run();
}

TEST_F(TestEdges, SourceComponentToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSourceComponent<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *node);
    node::make_edge(*node, *sink);

    node->run();
}

TEST_F(TestEdges, SourceToQueueToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *queue);
    node::make_edge(*queue, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToQueueToNodeToSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink   = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *queue);
    node::make_edge(*queue, *node);
    node::make_edge(*node, *sink);

    source->run();
    node->run();
    sink->run();
}

TEST_F(TestEdges, SourceToQueueToMultiSink)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *queue);
    node::make_edge(*queue, *sink1);
    node::make_edge(*queue, *sink2);

    source->run();
    sink1->run();
    sink2->run();
}

TEST_F(TestEdges, SourceToQueueToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto queue  = std::make_shared<node::TestQueue<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto node   = std::make_shared<node::TestNode<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *queue);
    node::make_edge(*queue, *sink1);
    node::make_edge(*queue, *node);
    node::make_edge(*node, *sink2);

    source->run();
    node->run();
    sink1->run();
    sink2->run();
}

TEST_F(TestEdges, SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestRouter<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *router);
    node::make_edge(*router->get_source("odd"), *sink1);
    node::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
    sink2->run();
}

TEST_F(TestEdges, SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource<int>>();
    auto router = std::make_shared<node::TestRouter<int>>();
    auto sink1  = std::make_shared<node::TestSink<int>>();
    auto sink2  = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *router);
    node::make_edge(*router->get_source("odd"), *sink1);
    node::make_edge(*router->get_source("even"), *sink2);

    source->run();
    sink1->run();
}

TEST_F(TestEdges, SourceToBroadcastToSink)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *broadcast);
    node::make_edge(*broadcast, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToBroadcastTypelessToSinkSinkFirst)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::BroadcastTypeless>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    node::make_edge(*broadcast, *sink);
    node::make_edge(*source, *broadcast);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToBroadcastTypelessToSinkSourceFirst)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::BroadcastTypeless>();
    auto sink      = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *broadcast);
    node::make_edge(*broadcast, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToMultipleBroadcastTypelessToSinkSinkFirst)
{
    auto source     = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink       = std::make_shared<node::TestSink<int>>();

    node::make_edge(*broadcast2, *sink);
    node::make_edge(*broadcast1, *broadcast2);
    node::make_edge(*source, *broadcast1);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToMultipleBroadcastTypelessToSinkSourceFirst)
{
    auto source     = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink       = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *broadcast1);
    node::make_edge(*broadcast1, *broadcast2);
    node::make_edge(*broadcast2, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, MultiSourceToMultipleBroadcastTypelessToMultiSink)
{
    auto source1    = std::make_shared<node::TestSource<int>>();
    auto source2    = std::make_shared<node::TestSource<int>>();
    auto broadcast1 = std::make_shared<node::BroadcastTypeless>();
    auto broadcast2 = std::make_shared<node::BroadcastTypeless>();
    auto sink1      = std::make_shared<node::TestSink<int>>();
    auto sink2      = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source1, *broadcast1);
    node::make_edge(*source2, *broadcast1);
    node::make_edge(*broadcast1, *broadcast2);
    node::make_edge(*broadcast2, *sink1);
    node::make_edge(*broadcast2, *sink2);

    source1->run();
    source2->run();
    sink1->run();
    sink2->run();
}

TEST_F(TestEdges, SourceToBroadcastToMultiSink)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSink<int>>();
    auto sink2     = std::make_shared<node::TestSink<int>>();

    node::make_edge(*source, *broadcast);
    node::make_edge(*broadcast, *sink1);
    node::make_edge(*broadcast, *sink2);

    source->run();
}

TEST_F(TestEdges, SourceToBroadcastToDifferentSinks)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSink<int>>();
    auto sink2     = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *broadcast);
    node::make_edge(*broadcast, *sink1);
    node::make_edge(*broadcast, *sink2);

    source->run();
}

TEST_F(TestEdges, SourceToBroadcastToSinkComponents)
{
    auto source    = std::make_shared<node::TestSource<int>>();
    auto broadcast = std::make_shared<node::Broadcast<int>>();
    auto sink1     = std::make_shared<node::TestSinkComponent<int>>();
    auto sink2     = std::make_shared<node::TestSinkComponent<int>>();

    node::make_edge(*source, *broadcast);
    node::make_edge(*broadcast, *sink1);
    node::make_edge(*broadcast, *sink2);

    source->run();
}

TEST_F(TestEdges, CombineLatest)
{
    auto source1 = std::make_shared<node::TestSource<int>>();
    auto source2 = std::make_shared<node::TestSource<float>>();

    auto combine_latest = std::make_shared<node::CombineLatest<int, float>>();

    auto sink = std::make_shared<node::TestSink<std::tuple<int, float>>>();

    node::make_edge(*source1, *combine_latest->get_sink<0>());
    node::make_edge(*source2, *combine_latest->get_sink<1>());
    node::make_edge(*combine_latest, *sink);

    source1->run();
    source2->run();

    sink->run();
}

}  // namespace mrc
