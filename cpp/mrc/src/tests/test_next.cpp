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

#include "tests/common.hpp"

#include "internal/runnable/runnable_resources.hpp"
#include "internal/system/system.hpp"
#include "internal/system/system_provider.hpp"
#include "internal/system/threading_resources.hpp"

#include "mrc/channel/ingress.hpp"
#include "mrc/data/reusable_pool.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/edge/edge_writable.hpp"
#include "mrc/node/generic_node.hpp"
#include "mrc/node/generic_sink.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/operators/conditional.hpp"
#include "mrc/node/readable_endpoint.hpp"
#include "mrc/node/rx_execute.hpp"
#include "mrc/node/rx_node.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/rx_subscribable.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/pipeline/segment.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runner.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/egress_ports.hpp"
#include "mrc/segment/object.hpp"
#include "mrc/segment/runnable.hpp"
#include "mrc/type_traits.hpp"
#include "mrc/utils/macros.hpp"

#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

using namespace mrc;

class TestNext : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_system_resources = std::make_unique<system::ThreadingResources>(
            system::SystemProvider(tests::make_system([](Options& options) {
                options.topology().user_cpuset("0-3");
                options.topology().restrict_gpus(true);
                options.engine_factories().set_engine_factory_options("thread_pool", [](EngineFactoryOptions& options) {
                    options.engine_type   = runnable::EngineType::Thread;
                    options.allow_overlap = false;
                    options.cpu_count     = 2;
                });
            })));

        m_resources = std::make_unique<runnable::RunnableResources>(*m_system_resources, 0);
    }

    void TearDown() override
    {
        m_resources.reset();
        m_system_resources.reset();
    }

    std::unique_ptr<system::ThreadingResources> m_system_resources;
    std::unique_ptr<runnable::RunnableResources> m_resources;
};

class ExGenSource : public node::GenericSource<int>
{
    void data_source(rxcpp::subscriber<int>& s) final {}
};

TEST_F(TestNext, LifeCycleSink)
{
    node::ReadableEndpoint<float> sink;
}

TEST_F(TestNext, LifeCycleSource)
{
    node::WritableEntrypoint<float> source;
}

TEST_F(TestNext, MakeEdgeSame)
{
    auto source = std::make_unique<node::WritableEntrypoint<float>>();
    auto sink   = std::make_unique<node::ReadableEndpoint<float>>();
    mrc::make_edge(*source, *sink);

    float input  = 3.14;
    float output = 0.0;

    source->await_write(input);
    sink->await_read(output);

    EXPECT_EQ(input, output);

    // Release the source first
    source.reset();
}

struct ExampleObject
{};

TEST_F(TestNext, UniqueToUnique)
{
    using input_t  = std::unique_ptr<ExampleObject>;
    using output_t = std::unique_ptr<ExampleObject>;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::ReadableEndpoint<output_t>>();

    mrc::make_edge(*source, *sink);

    input_t input   = std::make_unique<ExampleObject>();
    output_t output = nullptr;

    void* input_addr = input.get();

    source->await_write(std::move(input));
    sink->await_read(output);

    void* output_addr = output.get();

    EXPECT_EQ(input_addr, output_addr);

    // Release the source first
    source.reset();
}

TEST_F(TestNext, UniqueToConstShared)
{
    using input_t  = std::unique_ptr<ExampleObject>;
    using output_t = std::shared_ptr<const ExampleObject>;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::ReadableEndpoint<output_t>>();

    mrc::make_edge(*source, *sink);

    input_t input   = std::make_unique<ExampleObject>();
    output_t output = nullptr;

    void* input_addr = input.get();

    source->await_write(std::move(input));
    sink->await_read(output);

    const void* output_addr = output.get();

    EXPECT_EQ(input_addr, output_addr);

    // Release the source first
    source.reset();
}

TEST_F(TestNext, MakeEdgeConvertible)
{
    using input_t  = double;
    using output_t = float;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::ReadableEndpoint<output_t>>();

    mrc::make_edge(*source, *sink);

    input_t input   = 3.14;
    output_t output = 0.0;

    source->await_write(input);
    sink->await_read(output);

    EXPECT_FLOAT_EQ(input, output);

    // Release the source first
    source.reset();
}

TEST_F(TestNext, MakeEdgeConvertibleFromSinkRx)
{
    using input_t  = double;
    using output_t = float;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::RxSink<output_t>>();

    // todo - generalize the method that accepts raw or smart ptrs
    mrc::make_edge(*source, *sink);

    input_t input       = 3.14;
    output_t output     = 0.0;
    std::size_t counter = 0;

    source->await_write(input);
    source.reset();

    sink->set_observer(rxcpp::make_observer_dynamic<output_t>([input, &counter](output_t output) {
        EXPECT_FLOAT_EQ(input, output);
        ++counter;
    }));

    auto stream = node::RxExecute(std::move(sink));
    stream.subscribe();

    EXPECT_EQ(counter, 1);

    // Release the source first
    source.reset();
}

TEST_F(TestNext, MakeEdgeConvertibleFromSinkRxRunnable)
{
    using input_t  = double;
    using output_t = float;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::RxSink<output_t>>();

    // todo - generalize the method that accepts raw or smart ptrs
    mrc::make_edge(*source, *sink);

    input_t input       = 3.14;
    output_t output     = 0.0;
    std::size_t counter = 0;

    source->await_write(input);
    source.reset();

    // sink->set_observer(rxcpp::make_observer_dynamic<output_t>([input, &counter](output_t output) {
    //     EXPECT_FLOAT_EQ(input, output);
    //     ++counter;
    // }));

    sink->set_observer([input, &counter](output_t output) {
        EXPECT_FLOAT_EQ(input, output);
        ++counter;
    });

    auto runner = m_resources->launch_control().prepare_launcher(std::move(sink))->ignition();
    runner->await_join();

    EXPECT_EQ(counter, 1);

    // Release the source first
    source.reset();
}

class Node : public mrc::node::GenericNode<int, double>
{
    void on_data(int&& input, rxcpp::subscriber<double>& subscriber) final {}
};

TEST_F(TestNext, Node)
{
    auto node = std::make_unique<Node>();
}

class ExampleGenericNode : public node::GenericNode<int, int>
{
    void on_data(int&& data, rxcpp::subscriber<int>& output) final
    {
        DVLOG(10) << runnable::Context::get_runtime_context().info() << " data: " << data;
        data *= 2;
        output.on_next(std::move(data));
    }
};

class ExampleGenericSink : public node::GenericSink<int>
{
  public:
    std::size_t counter() const
    {
        return m_counter;
    }

  private:
    void on_data(int&& data) final
    {
        DVLOG(10) << runnable::Context::get_runtime_context().info() << " data: " << data;
        ++m_counter;
    }

    std::atomic<std::size_t> m_counter;
};

TEST_F(TestNext, GenericNodeAndSink)
{
    using input_t  = int;
    using output_t = int;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto node   = std::make_unique<ExampleGenericNode>();
    auto sink   = std::make_unique<ExampleGenericSink>();

    // todo - generalize the method that accepts raw or smart ptrs
    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    input_t input = 42;

    source->await_write(input);
    source->await_write(input);
    source->await_write(input);
    source.reset();

    auto runner_node = m_resources->launch_control().prepare_launcher(std::move(node))->ignition();
    auto runner_sink = m_resources->launch_control().prepare_launcher(std::move(sink))->ignition();

    // auto runner_node   = runnable::make_runner(std::move(node));
    // auto runner_sink   = runnable::make_runner(std::move(sink));
    // auto pool          = m_fiber_pool_mgr->make_pool(CpuSet("0"));
    // auto launcher_node = std::make_shared<runnable::FiberEngines>(pool);
    // auto launcher_sink = std::make_shared<runnable::FiberEngines>(pool);

    runner_node->await_join();
    runner_sink->await_join();

    const auto& const_sink = runner_sink->runnable_as<ExampleGenericSink>();
    EXPECT_EQ(const_sink.counter(), 3);
}

TEST_F(TestNext, ConcurrentSinkRxRunnable)
{
    using input_t  = double;
    using output_t = float;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto sink   = std::make_unique<node::RxSink<output_t>>();

    // todo - generalize the method that accepts raw or smart ptrs
    mrc::make_edge(*source, *sink);

    input_t input                      = 3.14;
    output_t output                    = 0.0;
    std::atomic<std::size_t> counter_0 = 0;
    std::atomic<std::size_t> counter_1 = 0;

    source->await_write(input);
    source->await_write(input);
    source->await_write(input);
    source->await_write(input);
    source->await_write(input);
    source.reset();

    sink->set_observer([input, &counter_0, &counter_1](output_t output) {
        const auto& ctx = runnable::Context::get_runtime_context();
        LOG(INFO) << ctx.rank() << ": sleep start";
        EXPECT_EQ(ctx.size(), 2);
        EXPECT_FLOAT_EQ(input, output);
        if (ctx.rank())  // NOLINT
        {
            ++counter_1;
        }
        else
        {
            ++counter_0;
        }
        auto time = (ctx.rank() == 0 ? 900 : 100);
        boost::this_fiber::sleep_for(std::chrono::milliseconds(time));
        LOG(INFO) << ctx.rank() << ": sleep end";
    });

    runnable::LaunchOptions options;
    options.pe_count = 2;

    auto runner = m_resources->launch_control().prepare_launcher(options, std::move(sink))->ignition();
    runner->await_join();

    EXPECT_EQ(counter_0, 1);
    EXPECT_EQ(counter_1, 4);
}

TEST_F(TestNext, SourceNodeSink)
{
    using input_t  = double;
    using output_t = float;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto node   = std::make_unique<node::RxNode<input_t, output_t>>(rxcpp::operators::map([](input_t d) -> output_t {
        return output_t(2.0 * d);
    }));
    auto sink   = std::make_unique<node::RxSink<output_t>>();

    mrc::make_edge(*source, *node);
    mrc::make_edge(*node, *sink);

    node->pipe(rxcpp::operators::map([](input_t i) {
        LOG(INFO) << "map: " << i;
        return static_cast<output_t>(i);
    }));
    sink->set_observer([](output_t o) {
        LOG(INFO) << "output: " << o;
    });

    auto runner_sink = m_resources->launch_control().prepare_launcher(std::move(sink))->ignition();
    auto runner_node = m_resources->launch_control().prepare_launcher(std::move(node))->ignition();

    source->await_write(3.14);
    source->await_write(42.);
    source->await_write(2.);
    source.reset();

    runner_node->await_join();
    runner_sink->await_join();
};

template <typename T, typename = void>
struct is_mrc_value : std::false_type
{};

template <typename T>
struct is_mrc_value<T, std::enable_if_t<std::is_arithmetic_v<T>>> : std::true_type
{};

template <typename T>
struct is_mrc_object
  : std::integral_constant<bool, std::is_class_v<T> and not is_mrc_value<T>::value and not is_smart_ptr<T>::value>
{};

template <typename T>
struct is_valid_node_type : std::integral_constant<bool, is_mrc_object<T>::value or is_mrc_value<T>::value>
{};

template <typename T, typename = void>
struct get_node_type  // NOLINT
{
    // static_assert(false, "an invalid node type");
};

template <typename T>
struct get_node_type<T, std::enable_if_t<is_mrc_value<T>::value>>
{
    using type = T;  // NOLINT
};

template <typename T>
struct get_node_type<T, std::enable_if_t<is_mrc_object<T>::value and !std::is_const_v<T>>>
{
    using type = std::unique_ptr<T>;  // NOLINT
};

template <typename T>
struct get_node_type<T, std::enable_if_t<is_mrc_object<T>::value and std::is_const_v<T>>>
{
    using type = std::shared_ptr<const T>;  // NOLINT
};

TEST_F(TestNext, TypeTraits)
{
    static_assert(is_valid_node_type<int>::value, " ");
    static_assert(is_valid_node_type<bool>::value, " ");
    static_assert(is_valid_node_type<float>::value, " ");
    static_assert(!is_valid_node_type<std::nullptr_t>::value, " ");
    static_assert(!is_valid_node_type<int*>::value, " ");
    static_assert(!is_valid_node_type<int&>::value, " ");

    static_assert(is_valid_node_type<ExampleObject>::value, " ");
    static_assert(!is_valid_node_type<ExampleObject*>::value, " ");
    static_assert(!is_valid_node_type<ExampleObject&>::value, " ");
    static_assert(!is_valid_node_type<std::unique_ptr<ExampleObject>>::value, " ");
    static_assert(!is_valid_node_type<std::shared_ptr<ExampleObject>>::value, " ");
    static_assert(!is_valid_node_type<std::shared_ptr<const ExampleObject>>::value, " ");

    static_assert(is_mrc_value<int>::value, " ");
    static_assert(is_mrc_value<const int>::value, " ");
    static_assert(!is_mrc_value<ExampleObject>::value, " ");

    static_assert(is_mrc_object<ExampleObject>::value, "should be true");
    static_assert(is_mrc_object<const ExampleObject>::value, "should be true");
    static_assert(!is_mrc_object<int>::value, "should be false");

    // get_node_type(T) => T if value otherwise unique_ptr<T>
    // get_node_type(const T) => const T if value otherwise shared_ptr<const T>
    static_assert(std::is_same_v<typename get_node_type<int>::type, int>, " ");
    static_assert(std::is_same_v<typename get_node_type<const int>::type, const int>, " ");
    static_assert(std::is_same_v<typename get_node_type<ExampleObject>::type, std::unique_ptr<ExampleObject>>, "true");
    static_assert(
        std::is_same_v<typename get_node_type<const ExampleObject>::type, std::shared_ptr<const ExampleObject>>,
        "true");
}

class MoveOnly
{
  public:
    DELETE_COPYABILITY(MoveOnly);
};

TEST_F(TestNext, TapUniquePtr)
{
    auto observable = rxcpp::observable<>::create<std::unique_ptr<int>>([](rxcpp::subscriber<std::unique_ptr<int>> s) {
                          s.on_next(std::make_unique<int>(1));
                          s.on_next(std::make_unique<int>(2));
                          s.on_completed();
                      })
                          .tap([](const std::unique_ptr<int>& int_ptr) {
                              LOG(INFO) << *int_ptr;
                          })
                          .map([](std::unique_ptr<int> i) {
                              *i = *i * 2;
                              return std::move(i);
                          });

    static_assert(rxcpp::detail::is_on_next_of<int, std::function<void(int)>>::value, " ");
    static_assert(rxcpp::detail::is_on_next_of<std::unique_ptr<int>, std::function<void(std::unique_ptr<int>)>>::value,
                  " ");

    auto observer = rxcpp::make_observer_dynamic<std::unique_ptr<int>>([](std::unique_ptr<int>&& int_ptr) {
        LOG(INFO) << "in observer: " << *int_ptr;
    });

    observable.subscribe([](std::unique_ptr<int> data) {
        LOG(INFO) << "in subscriber: " << *data;
    });
    observable.subscribe(observer);
}

TEST_F(TestNext, RxWithReusableOnNextAndOnError)
{
    auto pool = data::ReusablePool<int>::create(32);

    EXPECT_EQ(pool->size(), 0);

    for (int i = 0; i < 10; i++)
    {
        pool->emplace(42);
    }

    using data_t = data::Reusable<int>;

    auto observable = rxcpp::observable<>::create<data_t>([pool](rxcpp::subscriber<data_t> s) {
        for (int i = 0; i < 100; i++)
        {
            auto item = pool->await_item();
            s.on_next(std::move(item));
        }
        s.on_completed();
    });

    static_assert(rxcpp::detail::is_on_next_of<data_t, std::function<void(data_t)>>::value, " ");
    static_assert(rxcpp::detail::is_on_next_of<data_t, std::function<void(data_t&&)>>::value, " ");

    auto observer = rxcpp::make_observer_dynamic<data_t>(
        [](data_t&& int_ptr) {
            EXPECT_EQ(*int_ptr, 42);
        },
        [](std::exception_ptr ptr) {
            std::rethrow_exception(ptr);
        });

    observable.subscribe(observer);
}

TEST_F(TestNext, TapValue)
{
    auto observable = rxcpp::observable<>::create<int>([](rxcpp::subscriber<int> s) {
                          s.on_next(1);
                          s.on_next(2);
                          s.on_completed();
                      })
                          .tap([](const int& i) {
                              LOG(INFO) << i;
                          })
                          .map([](int i) {
                              i = i * 2;
                              return i;
                          });

    auto observer = rxcpp::make_observer_dynamic<int>([](int i) {
        LOG(INFO) << "in observer: " << i;
    });

    observable.subscribe([](int i) {
        LOG(INFO) << "in subscriber: " << i;
    });
    observable.subscribe(observer);
}

enum class Routes
{
    Even,
    Odd
};

inline std::ostream& operator<<(std::ostream& os, const Routes cat)
{
    switch (cat)
    {
    case Routes::Even:
        os << "Even";
        break;
    case Routes::Odd:
        os << "Odd";
        break;
    }
    return os;
};

TEST_F(TestNext, Conditional)
{
    using input_t  = int;
    using output_t = int;

    auto source = std::make_unique<node::WritableEntrypoint<input_t>>();
    auto even   = std::make_unique<node::ReadableEndpoint<output_t>>();
    auto odd    = std::make_unique<node::ReadableEndpoint<output_t>>();

    auto cond = std::make_shared<node::Conditional<Routes, input_t>>([](const input_t& i) -> Routes {
        if (i % 2 == 0)
        {
            return Routes::Even;
        }
        return Routes::Odd;
    });

    // make edge via pipe fn
    mrc::make_edge(*source, *cond);
    mrc::make_edge(*cond->get_source(Routes::Odd), *odd);
    mrc::make_edge(*cond->get_source(Routes::Even), *even);

    source->await_write(0);
    source->await_write(1);
    source->await_write(2);
    source.reset();

    output_t output;

    even->await_read(output);
    EXPECT_EQ(output, 0);
    even->await_read(output);
    EXPECT_EQ(output, 2);
    odd->await_read(output);
    EXPECT_EQ(output, 1);
}

class PrivateSource : private node::SourceChannelOwner<int>
{
  public:
    node::SourceChannelOwner<int>& source()
    {
        return *this;
    }
};

// class PrivateSink : private node::RxSink<int>
// {
//   public:
//     node::SinkChannelOwner<int>& sink()
//     {
//         return *this;
//     }
// };

/*
TEST_F(TestNext, PrivateInheritance)
{
    auto source = std::make_unique<PrivateSource>();
    auto sink   = std::make_unique<PrivateSink>();

    node::EdgeBuilder::make_edge(source->source(), sink->sink());
}
*/

TEST_F(TestNext, Segment)
{
    auto segment = Segment::create("test", segment::EgressPorts<int>({"test"}), [](segment::IBuilder& s) {});

    segment::EgressPorts<int> ports({"test"});

    // segment::Segment seg("test", ports, [](segment::IBuilder& builder) {});
}

TEST_F(TestNext, SegmentRunnable)
{
    auto node = std::make_shared<segment::Runnable<ExGenSource>>();

    EXPECT_TRUE(node->is_source());
    EXPECT_FALSE(node->is_sink());

    auto sink = std::make_unique<node::ReadableEndpoint<int>>();

    mrc::make_edge(node->object(), *sink);

    // Release the node first
    node.reset();
}

TEST_F(TestNext, SegmentBuilder)
{
    auto init = [](segment::IBuilder& segment) {
        auto src = segment.make_source<std::string>("x_src", [&](rxcpp::subscriber<std::string> s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        });

        auto x = segment.make_node<std::string, std::string>("x");

        auto y = segment.make_node<std::string, double>("y", rxcpp::operators::map([](std::string s) -> double {
                                                            return 1.0;
                                                        }));

        auto z = segment.make_sink<double>("z", [](double d) {
            LOG(INFO) << d;
        });

        EXPECT_TRUE(src->is_runnable());
        EXPECT_TRUE(x->is_runnable());
        EXPECT_TRUE(y->is_runnable());
        EXPECT_TRUE(z->is_runnable());

        segment.make_edge(src, x);
        segment.make_edge(x, y);
        segment.make_edge(y, z);
    };

    auto definition = Segment::create("segment_test", init);
}
