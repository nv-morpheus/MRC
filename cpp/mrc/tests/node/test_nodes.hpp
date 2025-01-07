/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "../test_mrc.hpp"  // IWYU pragma: associated

#include "mrc/channel/buffered_channel.hpp"  // IWYU pragma: keep
#include "mrc/channel/forward.hpp"
#include "mrc/edge/edge.hpp"  // for Edge
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_channel.hpp"
#include "mrc/edge/edge_holder.hpp"  // for EdgeHolder
#include "mrc/edge/edge_readable.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/exceptions/runtime_error.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/operators/broadcast.hpp"
#include "mrc/node/operators/combine_latest.hpp"
#include "mrc/node/operators/node_component.hpp"
#include "mrc/node/operators/round_robin_router_typeless.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/operators/with_latest_from.hpp"
#include "mrc/node/operators/zip.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/source_properties.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>  // for observable_member

#include <deque>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

// IWYU pragma: no_forward_declare mrc::channel::Channel

// IWYU thinks we need vector for make_segment
// IWYU pragma: no_include <vector>

using namespace std::chrono_literals;

namespace mrc::node {

template <size_t N, typename... T>
typename std::enable_if<(N >= sizeof...(T))>::type print_tuple(std::ostream& /*unused*/,
                                                               const std::tuple<T...>& /*unused*/)
{}

template <size_t N, typename... T>
typename std::enable_if<(N < sizeof...(T))>::type print_tuple(std::ostream& os, const std::tuple<T...>& tup)
{
    if (N != 0)
    {
        os << ", ";
    }
    os << std::get<N>(tup);
    print_tuple<N + 1>(os, tup);
}

// Utility function to print tuples
template <typename... T>
std::ostream& operator<<(std::ostream& os, const std::tuple<T...>& tup)
{
    os << "[";
    print_tuple<0>(os, tup);
    return os << "]";
}

template <typename T>
class EdgeReadableLambda : public edge::IEdgeReadable<T>
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

    channel::Status await_read_until(T& t, const mrc::channel::time_point_t& tp) override
    {
        throw std::runtime_error("Not implemented");
        return channel::Status::error;
    }

  private:
    std::function<channel::Status(T&)> m_on_await_read;
    std::function<void()> m_on_complete;
};

template <typename T>
class EdgeWritableLambda : public edge::IEdgeWritable<T>
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
class TestSource : public WritableAcceptor<T>, public ReadableProvider<T>, public SourceChannelOwner<T>
{
  public:
    TestSource(std::vector<T> values) :
      m_init_values(values),
      m_values(std::deque<T>(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end())))
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    TestSource(std::initializer_list<T> values) :
      TestSource(std::vector<T>(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end())))
    {}

    TestSource(size_t count) : TestSource(gen_values(count)) {}

    TestSource() : TestSource(3) {}

    void run()
    {
        // Just push them all
        this->push(m_values.size());
    }

    void push_one()
    {
        this->push(1);
    }

    void push(size_t count = 1)
    {
        auto output = this->get_writable_edge();

        for (size_t i = 0; i < count; ++i)
        {
            if (output->await_write(std::move(m_values.front())) != channel::Status::success)
            {
                this->release_edge_connection();
                throw exceptions::MrcRuntimeError("Failed to push values. await_write returned non-success status");
            }

            m_values.pop();
        }

        if (m_values.empty())
        {
            this->release_edge_connection();
        }
    }

    const std::vector<T>& get_init_values()
    {
        return m_init_values;
    }

  private:
    static std::vector<T> gen_values(size_t count)
    {
        std::vector<T> values;

        for (size_t i = 0; i < count; ++i)
        {
            values.emplace_back(i);
        }

        return values;
    }

    std::vector<T> m_init_values;
    std::queue<T> m_values;
};

template <typename T>
class TestNode : public WritableProvider<T>,
                 public ReadableAcceptor<T>,
                 public WritableAcceptor<T>,
                 public ReadableProvider<T>,
                 public SinkChannelOwner<T>,
                 public SourceChannelOwner<T>
{
  public:
    TestNode()
    {
        SinkChannelOwner<T>::set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
        SourceChannelOwner<T>::set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void run()
    {
        auto input  = this->get_readable_edge();
        auto output = this->get_writable_edge();

        int t;

        while (input->await_read(t) == channel::Status::success)
        {
            VLOG(10) << "Node got value: " << t;

            if (output->await_write(std::move(t)) != channel::Status::success)
            {
                SinkChannelOwner<T>::release_edge_connection();
                SourceChannelOwner<T>::release_edge_connection();
                throw exceptions::MrcRuntimeError("Failed to push values. await_write returned non-success status");
            }
        }

        VLOG(10) << "Node exited run";

        SinkChannelOwner<T>::release_edge_connection();
        SourceChannelOwner<T>::release_edge_connection();
    }
};

template <typename T>
class TestSink : public WritableProvider<T>, public ReadableAcceptor<T>, public SinkChannelOwner<T>
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
            m_values.emplace_back(std::move(t));
        }

        VLOG(10) << "Sink exited run";

        this->release_edge_connection();
    }

    const std::vector<T>& get_values()
    {
        return m_values;
    }

  private:
    std::vector<T> m_values;
};

template <typename T>
class TestQueue : public WritableProvider<T>, public ReadableProvider<T>
{
  public:
    TestQueue()
    {
        this->set_channel(std::make_unique<mrc::channel::BufferedChannel<T>>());
    }

    void set_channel(std::unique_ptr<mrc::channel::Channel<T>> channel)
    {
        edge::EdgeChannel<T> edge_channel(std::move(channel));

        SinkProperties<T>::init_owned_edge(edge_channel.get_writer());
        SourceProperties<T>::init_owned_edge(edge_channel.get_reader());
    }
};

template <typename T>
class TestSourceComponent : public GenericSourceComponent<T>
{
  public:
    TestSourceComponent(std::vector<T> values) :
      m_init_values(values),
      m_values(std::deque<T>(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end())))
    {}

    TestSourceComponent(std::initializer_list<T> values) :
      TestSourceComponent(
          std::vector<T>(std::make_move_iterator(values.begin()), std::make_move_iterator(values.end())))
    {}

    TestSourceComponent(size_t count) : TestSourceComponent(gen_values(count)) {}

    TestSourceComponent() : TestSourceComponent(3) {}

    const std::vector<T>& get_init_values()
    {
        return m_init_values;
    }

  protected:
    channel::Status get_data(T& data) override
    {
        // Close after all values have been pulled
        if (m_values.empty())
        {
            return channel::Status::closed;
        }

        data = std::move(m_values.front());
        m_values.pop();

        VLOG(10) << "TestSourceComponent emmitted value: " << data;

        return channel::Status::success;
    }

    void on_complete() override
    {
        VLOG(10) << "TestSourceComponent completed";
    }

  private:
    static std::vector<T> gen_values(size_t count)
    {
        std::vector<T> values;

        for (size_t i = 0; i < count; ++i)
        {
            values.emplace_back(i);
        }

        return values;
    }

    std::vector<T> m_init_values;
    std::queue<T> m_values;
};

template <typename T>
class TestNodeComponent : public NodeComponent<T, T>
{
  public:
    TestNodeComponent() = default;

    ~TestNodeComponent() override
    {
        // Debug print
        VLOG(10) << "Destroying TestNodeComponent";
    }

    channel::Status on_next(int&& t) override
    {
        VLOG(10) << "TestNodeComponent got value: " << t;

        return this->get_writable_edge()->await_write(t);
    }

    void do_on_complete() override
    {
        VLOG(10) << "TestSinkComponent completed";
    }
};

template <typename T>
class TestRxNodeComponent : public RxNodeComponent<T, T>
{
    using base_t = node::RxNodeComponent<T, T>;

  public:
    using typename base_t::stream_fn_t;

    void make_stream(stream_fn_t fn)
    {
        return base_t::make_stream([this, fn](auto&&... args) {
            stream_fn_called = true;
            return fn(std::forward<decltype(args)>(args)...);
        });
    }

    bool stream_fn_called = false;
};

template <typename T>
class TestSinkComponent : public WritableProvider<T>
{
  public:
    TestSinkComponent()
    {
        this->init_owned_edge(std::make_shared<EdgeWritableLambda<T>>(
            [this](T&& t) {
                // Call this object
                return this->await_write(std::move(t));
            },
            [this]() {
                this->on_complete();
            }));
    }

    const std::vector<T>& get_values()
    {
        return m_values;
    }

  protected:
    channel::Status await_write(T&& t)
    {
        VLOG(10) << "TestSinkComponent got value: " << t;

        m_values.emplace_back(std::move(t));

        return channel::Status::success;
    }

    void on_complete()
    {
        VLOG(10) << "TestSinkComponent completed";
    }

  private:
    std::vector<T> m_values;
};

class TestRouter : public StaticRouterComponentBase<std::string, int>
{
  public:
    TestRouter() : StaticRouterComponentBase<std::string, int>(std::vector<std::string>{"odd", "even"}) {}

  protected:
    std::string determine_key_for_value(const int& t) override
    {
        return t % 2 == 1 ? "odd" : "even";
    }
};

template <typename T>
class TestDynamicRouter : public DynamicRouterComponentBase<std::string, T>
{
  protected:
    std::string determine_key_for_value(const T& t) override
    {
        auto keys = DynamicRouterComponentBase<std::string, T>::edge_connection_keys();

        return keys[t % keys.size()];
    }
};
}  // namespace mrc::node
