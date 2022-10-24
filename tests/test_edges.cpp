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

#include "test_srf.hpp"  // IWYU pragma: associated

#include "srf/channel/buffered_channel.hpp"
#include "srf/channel/status.hpp"
#include "srf/core/addresses.hpp"
#include "srf/core/executor.hpp"
#include "srf/node/channel_holder.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/rx_subscribable.hpp"
#include "srf/node/source_properties.hpp"
#include "srf/options/options.hpp"
#include "srf/options/placement.hpp"
#include "srf/options/topology.hpp"
#include "srf/pipeline/pipeline.hpp"
#include "srf/segment/builder.hpp"
#include "srf/types.hpp"

#include <glog/logging.h>
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

namespace srf::node {

template <typename T>
class EdgeReadableLambda : public EdgeReadable<T>
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

    virtual channel::Status await_read(T& t)
    {
        return m_on_await_read(t);
    }

  private:
    std::function<channel::Status(T&)> m_on_await_read;
    std::function<void()> m_on_complete;
};

template <typename T>
class EdgeWritableLambda : public EdgeWritable<T>
{
  public:
    EdgeWritableLambda(std::function<channel::Status(T&&)>&& on_await_write,
                       std::function<void()>&& on_complete = nullptr) :
      m_on_await_write(std::move(on_await_write)),
      m_on_complete(std::move(on_complete))
    {}

    ~EdgeWritableLambda()
    {
        if (m_on_complete)
        {
            m_on_complete();
        }
    }

    virtual channel::Status await_write(T&& t)
    {
        return m_on_await_write(std::move(t));
    }

  private:
    std::function<channel::Status(T&&)> m_on_await_write;
    std::function<void()> m_on_complete;
};

class TestSource : public IngressAcceptor<int>, public EgressProvider<int>, public SourceChannel<int>
{
  public:
    TestSource()
    {
        this->set_channel(std::make_unique<srf::channel::BufferedChannel<int>>());
    }

    void run()
    {
        auto output = this->get_writable_edge();

        for (int i = 0; i < 3; i++)
        {
            if (output->await_write(int(i)) != channel::Status::success)
            {
                break;
            }
        }

        this->release_edge();
    }
};

class TestNode : public IngressProvider<int>,
                 public EgressAcceptor<int>,
                 public IngressAcceptor<int>,
                 public EgressProvider<int>,
                 public SinkChannel<int>,
                 public SourceChannel<int>
{
  public:
    TestNode()
    {
        this->set_channel(std::make_unique<srf::channel::BufferedChannel<int>>());
    }

    void set_channel(std::unique_ptr<srf::channel::Channel<int>> channel)
    {
        EdgeChannel<int> edge_channel(std::move(channel));

        SinkChannel<int>::do_set_channel(edge_channel);
        SourceChannel<int>::do_set_channel(edge_channel);
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

        SinkChannel<int>::release_edge();
        SourceChannel<int>::release_edge();
    }
};

class TestSink : public IngressProvider<int>, public EgressAcceptor<int>, public SinkChannel<int>
{
  public:
    TestSink()
    {
        this->set_channel(std::make_unique<srf::channel::BufferedChannel<int>>());
    }

    void run()
    {
        auto input = this->get_readable_edge();

        int t;

        while (input->await_read(t) == channel::Status::success)
        {
            VLOG(10) << "Sink got value: " << t;
        }

        VLOG(10) << "Sink exited run";
    }
};

class TestQueue : public IngressProvider<int>, public EgressProvider<int>
{
  public:
    TestQueue()
    {
        this->set_channel(std::make_unique<srf::channel::BufferedChannel<int>>());
    }

    void set_channel(std::unique_ptr<srf::channel::Channel<int>> channel)
    {
        std::shared_ptr<srf::channel::Channel<int>> shared_channel = std::move(channel);

        // Create a new edge that will close the channel when either side disconnects
        auto channel_reader = std::make_shared<EdgeChannelReader<int>>(shared_channel);
        auto channel_writer = std::make_shared<EdgeChannelWriter<int>>(shared_channel);

        SinkProperties<int>::init_edge(channel_writer);
        SourceProperties<int>::init_edge(channel_reader);
    }
};

class TestSourceComponent : public EgressProvider<int>
{
  public:
    TestSourceComponent()
    {
        this->init_edge(std::make_shared<EdgeReadableLambda<int>>(
            [this](int& t) {
                // Call this object
                return this->await_read(t);
            },
            [this]() { this->on_complete(); }));
    }

    channel::Status await_read(int& t)
    {
        VLOG(10) << "TestSourceComponent got value: " << t;

        t = 1;

        return channel::Status::success;
    }

    void on_complete()
    {
        VLOG(10) << "TestSourceComponent completed";
    }
};

class TestNodeComponent : public IngressProvider<int>, public IngressAcceptor<int>
{
  public:
    TestNodeComponent()
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
                SourceProperties<int>::release_edge();
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

class TestSinkComponent : public IngressProvider<int>
{
  public:
    TestSinkComponent()
    {
        this->init_edge(std::make_shared<EdgeWritableLambda<int>>(
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

class TestRouter : public IngressProvider<int>
{
    class UpstreamEdge : public EdgeWritable<int>, public MultiSourceProperties<int, std::string>
    {
      public:
        UpstreamEdge(TestRouter& parent) : m_parent(parent) {}

        ~UpstreamEdge() override
        {
            m_parent.on_complete();
        }

        channel::Status await_write(int&& t) override
        {
            VLOG(10) << "TestRouter got value: " << t;

            // Determine key for value
            auto key = m_parent.determine_key_for_value(t);

            return this->get_writable_edge(key)->await_write(std::move(t));
        }

        void add_downstream(std::string key, std::shared_ptr<EdgeWritable<int>> downstream)
        {
            this->set_edge(std::move(key), std::move(downstream));
        }

      private:
        TestRouter& m_parent;
    };

    class DownstreamEdge : public IIngressAcceptor<int>
    {
      public:
        DownstreamEdge(std::weak_ptr<UpstreamEdge> upstream, std::string key) :
          m_upstream(std::move(upstream)),
          m_key(std::move(key))
        {}

        void set_ingress(std::shared_ptr<EdgeWritable<int>> ingress) override
        {
            // Get a lock to the upstream edge
            if (auto upstream = m_upstream.lock())
            {
                upstream->add_downstream(m_key, ingress);
            }
            else
            {
                LOG(ERROR) << "Could not set ingress to upstream edge. Upstream edge has been destroyed.";
            }
        }

      private:
        std::weak_ptr<UpstreamEdge> m_upstream;
        std::string m_key;
    };

  public:
    TestRouter()
    {
        auto upstream = std::make_shared<UpstreamEdge>(*this);

        // Save it to avoid casting
        m_upstream = upstream;

        IngressProvider<int>::init_edge(upstream);
    }

    std::shared_ptr<IIngressAcceptor<int>> get_source(const std::string& key) const
    {
        auto found = m_downstream.find(key);

        if (found == m_downstream.end())
        {
            auto new_source = std::make_shared<DownstreamEdge>(m_upstream, key);

            const_cast<TestRouter*>(this)->m_downstream[key] = new_source;

            return new_source;
        }

        return found->second;
    }

    void on_complete()
    {
        VLOG(10) << "TestRouter completed";
    }

  protected:
    std::string determine_key_for_value(const int& t)
    {
        return "test";
    }

  private:
    std::weak_ptr<UpstreamEdge> m_upstream;
    std::map<std::string, std::shared_ptr<DownstreamEdge>> m_downstream;
};

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
                SourceProperties<int>::release_edge();
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

class TestBroadcast : public IngressProvider<int>, public IIngressAcceptor<int>
{
    class BroadcastEdge : public EdgeWritable<int>, public MultiSourceProperties<int, size_t>
    {
      public:
        BroadcastEdge(TestBroadcast& parent) : m_parent(parent) {}

        ~BroadcastEdge()
        {
            m_parent.on_complete();
        }

        channel::Status await_write(int&& t) override
        {
            VLOG(10) << "BroadcastEdge got value: " << t;

            for (size_t i = this->edge_count() - 1; i > 0; i--)
            {
                // Make a copy
                int x = t;

                auto response = this->get_writable_edge(i)->await_write(std::move(x));

                if (response != channel::Status::success)
                {
                    return response;
                }
            }

            // Write index 0 last
            return this->get_writable_edge(0)->await_write(std::move(t));
        }

        void add_downstream(std::shared_ptr<EdgeWritable<int>> downstream)
        {
            auto edge_count = this->edge_count();

            this->set_edge(edge_count, downstream);
        }

      private:
        TestBroadcast& m_parent;
        // std::vector<std::shared_ptr<EdgeWritable<int>>> m_outputs;
    };

  public:
    TestBroadcast()
    {
        auto edge = std::make_shared<BroadcastEdge>(*this);

        // Save to avoid casting
        m_edge = edge;

        IngressProvider<int>::init_edge(edge);
    }

    ~TestBroadcast()
    {
        // Debug print
        VLOG(10) << "Destroying TestBroadcast";
    }

    void set_ingress(std::shared_ptr<EdgeWritable<int>> ingress) override
    {
        if (auto e = m_edge.lock())
        {
            e->add_downstream(ingress);
        }
        else
        {
            LOG(ERROR) << "Edge was destroyed";
        }
    }

    void on_complete()
    {
        VLOG(10) << "TestBroadcast completed";
    }

  private:
    std::weak_ptr<BroadcastEdge> m_edge;
};

}  // namespace srf::node

namespace srf {

TEST_F(TestEdges, SourceToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToNodeToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto node   = std::make_shared<node::TestNode>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceToNodeToNodeToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto node1  = std::make_shared<node::TestNode>();
    auto node2  = std::make_shared<node::TestNode>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *node1);
    node::make_edge2(*node1, *node2);
    node::make_edge2(*node2, *sink);
}

TEST_F(TestEdges, SourceToSinkMultiFail)
{
    auto source = std::make_shared<node::TestSource>();
    auto sink1  = std::make_shared<node::TestSink>();
    auto sink2  = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *sink1);
    EXPECT_THROW(node::make_edge2(*source, *sink2), std::runtime_error);
}

TEST_F(TestEdges, SourceToSinkComponent)
{
    auto source = std::make_shared<node::TestSource>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *sink);
}

TEST_F(TestEdges, SourceComponentToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *sink);
}

TEST_F(TestEdges, SourceComponentToNodeToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto node   = std::make_shared<node::TestNodeComponent>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceToNodeComponentToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto node   = std::make_shared<node::TestNodeComponent>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSource>();
    auto node   = std::make_shared<node::TestNodeComponent>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceToNodeComponentToSinkComponent)
{
    auto source = std::make_shared<node::TestSource>();
    auto node   = std::make_shared<node::TestNodeComponent>();
    auto sink   = std::make_shared<node::TestSinkComponent>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceComponentToNodeToSinkComponent)
{
    auto source = std::make_shared<node::TestSourceComponent>();
    auto node   = std::make_shared<node::TestNode>();
    auto sink   = std::make_shared<node::TestSinkComponent>();

    node::make_edge2(*source, *node);
    node::make_edge2(*node, *sink);
}

TEST_F(TestEdges, SourceToQueueToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto queue  = std::make_shared<node::TestQueue>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *queue);
    node::make_edge2(*queue, *sink);

    source->run();
    sink->run();
}

TEST_F(TestEdges, SourceToQueueToNodeToSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto queue  = std::make_shared<node::TestQueue>();
    auto node   = std::make_shared<node::TestNode>();
    auto sink   = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *queue);
    node::make_edge2(*queue, *node);
    node::make_edge2(*node, *sink);

    source->run();
    node->run();
    sink->run();
}

TEST_F(TestEdges, SourceToQueueToMultiSink)
{
    auto source = std::make_shared<node::TestSource>();
    auto queue  = std::make_shared<node::TestQueue>();
    auto sink1  = std::make_shared<node::TestSink>();
    auto sink2  = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *queue);
    node::make_edge2(*queue, *sink1);
    node::make_edge2(*queue, *sink2);
}

TEST_F(TestEdges, SourceToQueueToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource>();
    auto queue  = std::make_shared<node::TestQueue>();
    auto sink1  = std::make_shared<node::TestSink>();
    auto node   = std::make_shared<node::TestNode>();
    auto sink2  = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *queue);
    node::make_edge2(*queue, *sink1);
    node::make_edge2(*queue, *node);
    node::make_edge2(*node, *sink2);
}

TEST_F(TestEdges, SourceToRouterToSinks)
{
    auto source = std::make_shared<node::TestSource>();
    auto router = std::make_shared<node::TestRouter>();
    auto sink1  = std::make_shared<node::TestSink>();
    auto sink2  = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *router);
    node::make_edge2(*router->get_source("odd"), *sink1);
    node::make_edge2(*router->get_source("even"), *sink2);
}

TEST_F(TestEdges, SourceToRouterToDifferentSinks)
{
    auto source = std::make_shared<node::TestSource>();
    auto router = std::make_shared<node::TestRouter>();
    auto sink1  = std::make_shared<node::TestSink>();
    auto sink2  = std::make_shared<node::TestSinkComponent>();

    node::make_edge2(*source, *router);
    node::make_edge2(*router->get_source("odd"), *sink1);
    node::make_edge2(*router->get_source("even"), *sink2);
}

TEST_F(TestEdges, SourceToBroadcastToSink)
{
    auto source    = std::make_shared<node::TestSource>();
    auto broadcast = std::make_shared<node::TestBroadcast>();
    auto sink      = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *broadcast);
    node::make_edge2(*broadcast, *sink);
}

TEST_F(TestEdges, SourceToBroadcastToMultiSink)
{
    auto source    = std::make_shared<node::TestSource>();
    auto broadcast = std::make_shared<node::TestBroadcast>();
    auto sink1     = std::make_shared<node::TestSink>();
    auto sink2     = std::make_shared<node::TestSink>();

    node::make_edge2(*source, *broadcast);
    node::make_edge2(*broadcast, *sink1);
    node::make_edge2(*broadcast, *sink2);
}

TEST_F(TestEdges, SourceToBroadcastToDifferentSinks)
{
    auto source    = std::make_shared<node::TestSource>();
    auto broadcast = std::make_shared<node::TestBroadcast>();
    auto sink1     = std::make_shared<node::TestSink>();
    auto sink2     = std::make_shared<node::TestSinkComponent>();

    node::make_edge2(*source, *broadcast);
    node::make_edge2(*broadcast, *sink1);
    node::make_edge2(*broadcast, *sink2);
}

TEST_F(TestEdges, SourceToBroadcastToSinkComponents)
{
    auto source    = std::make_shared<node::TestSource>();
    auto broadcast = std::make_shared<node::TestBroadcast>();
    auto sink1     = std::make_shared<node::TestSinkComponent>();
    auto sink2     = std::make_shared<node::TestSinkComponent>();

    node::make_edge2(*source, *broadcast);
    node::make_edge2(*broadcast, *sink1);
    node::make_edge2(*broadcast, *sink2);

    source->run();
}

}  // namespace srf
