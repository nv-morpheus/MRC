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

#include "test_segment.hpp"  // IWYU pragma: associated
#include "test_srf.hpp"      // IWYU pragma: associated

#include <srf/channel/status.hpp>

#include <srf/core/egress_node.hpp>
#include <srf/core/executor.hpp>
#include <srf/core/ingress_node.hpp>
#include <srf/core/utils.hpp>
#include <srf/pipeline/pipeline.hpp>

#include <srf/segment/egress_ports.hpp>
#include <srf/segment/ingress_ports.hpp>
#include <srf/segment/segment.hpp>
#include <srf/types.hpp>

#include <rxcpp/operators/rx-map.hpp>
#include <rxcpp/operators/rx-tap.hpp>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx-observable.hpp>
#include <rxcpp/rx-observer.hpp>
#include <rxcpp/rx-predef.hpp>
#include <rxcpp/rx-subscriber.hpp>
#include <rxcpp/sources/rx-iterate.hpp>

#include <array>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

// IWYU thinks we need a few std headers for rxcpp
// IWYU pragma: no_include <exception>
// IWYU pragma: no_include <map>
// IWYU pragma: no_include <vector>

class SegmentComponentTests : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline  = pipeline::make_pipeline();
        m_resources = std::make_shared<TestSegmentResources>(15);
    }

    void TearDown() override{};

    std::function<void(Segment&)> m_initializer = [](Segment&) {};
    std::unique_ptr<Pipeline> m_pipeline;
    std::shared_ptr<TestSegmentResources> m_resources;

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_get_ingress(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            auto ingress = seg_node->get_ingress();
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_ingress on a node which is not backed by an IngressPort."))
            {
                return false;
            }

            // Unexpected error
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_get_egress(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            auto egress = seg_node->get_egress();
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_egress on a node which is not backed by an EgressPort."))
            {
                return false;
            }
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_register_port(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            seg_node->register_port(m_resources->port_registry(), 0, 0, 0);
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to register_port on a node that is not an Ingress or Egress."))
            {
                return false;
            }

            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_rx_node_upcast(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            if (std::dynamic_pointer_cast<rx::RxNode<SourceTypeT, SinkTypeT>>(seg_node->get_segment_object()) ==
                nullptr)
            {
                return false;
            }
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_segment_object on a node which is not backed by an RxNode."))
            {
                return false;
            }
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_rx_source_upcast(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            if (std::dynamic_pointer_cast<rx::RxSource<SourceTypeT>>(seg_node->get_segment_object()) == nullptr)
            {
                return false;
            }
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_rx_source on a node which is not backed by an RxSource."))
            {
                return false;
            }
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_rx_sink_upcast(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            if (std::dynamic_pointer_cast<rx::RxSink<SinkTypeT>>(seg_node->get_segment_object()) == nullptr)
            {
                return false;
            }
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_rx_sink on a node which is not backed by an RxSink."))
            {
                return false;
            }
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_source_upcast(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            if (seg_node->get_segment_source() == nullptr)
            {
                return false;
            }
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_source on a node which is not backed by a SegmentSource."))
            {
                return false;
            }
            throw;
        }
    }

    template <typename SourceTypeT, typename SinkTypeT>
    bool check_sink_upcast(std::shared_ptr<SegmentNodeInfo<SourceTypeT, SinkTypeT>> seg_node)
    {
        try
        {
            if (seg_node->get_segment_sink() == nullptr)
            {
                return false;
            }
            return true;
        } catch (std::runtime_error e)
        {
            if (e.what() == std::string("Call to get_sink on a node which is not backed by a SegmentSink."))
            {
                return false;
            }
            throw;
        }
    }
};

TEST(SegmentComponents, IngressPortsCreation)
{
    using p1_type_t = IngressPorts<int>;
    using p2_type_t = IngressPorts<int, int, double, float, std::string, std::stringstream>;

    auto p  = p1_type_t({"an_int"});
    auto p2 = p2_type_t({"an_int", "another_int", "a_double", "a_float", "a_string", "a_sstream"});

    EXPECT_EQ(p1_type_t::Size, 1);
    EXPECT_EQ(p1_type_t::Size, p.m_port_names.size());
    EXPECT_EQ(typeid(p1_type_t::nth_t<0>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(p.m_port_names[0], "an_int");

    EXPECT_EQ(p2_type_t::Size, 6);
    EXPECT_EQ(p2_type_t::Size, p2.m_port_names.size());
    EXPECT_EQ(typeid(p2_type_t::nth_t<0>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<1>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<2>).hash_code(), typeid(double).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<3>).hash_code(), typeid(float).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<4>).hash_code(), typeid(std::string).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<5>).hash_code(), typeid(std::stringstream).hash_code());
    EXPECT_EQ(p2.m_port_names[0], "an_int");
    EXPECT_EQ(p2.m_port_names[1], "another_int");
    EXPECT_EQ(p2.m_port_names[2], "a_double");
    EXPECT_EQ(p2.m_port_names[3], "a_float");
    EXPECT_EQ(p2.m_port_names[4], "a_string");
    EXPECT_EQ(p2.m_port_names[5], "a_sstream");
}

TEST(SegmentComponents, IngressPortsDuplicateFail)
{
    using p1_type_t = IngressPorts<int, int, double, float, std::string, std::stringstream>;
    try
    {
        auto p1 = p1_type_t({"an_int", "another_int", "an_int", "a_float", "a_string", "a_sstream"});
        FAIL() << "Expected std::invalid_argument";
    } catch (std::invalid_argument e)
    {
        EXPECT_EQ(e.what(), std::string("Ingress port names must be unique: 'an_int'"));
    } catch (...)
    {
        FAIL() << "Expected std::invalid_argument";
    }
}

TEST(SegmentComponents, EgressPortsDuplicateFail)
{
    using p1_type_t = EgressPorts<int, int, double, float, std::string, std::stringstream>;
    try
    {
        auto p1 = p1_type_t({"an_int", "another_int", "an_int", "a_float", "a_string", "a_sstream"});
        FAIL() << "Expected std::invalid_argument";
    } catch (std::invalid_argument e)
    {
        EXPECT_EQ(e.what(), std::string("Egress port names must be unique: 'an_int'"));
    } catch (...)
    {
        FAIL() << "Expected std::invalid_argument";
    }
}

TEST(SegmentComponents, EgressPortsCreation)
{
    using p1_type_t = EgressPorts<int>;
    using p2_type_t = EgressPorts<int, int, double, float, std::string, std::stringstream>;

    auto p  = p1_type_t({"an_int"});
    auto p2 = p2_type_t({"an_int", "another_int", "a_double", "a_float", "a_string", "a_sstream"});

    EXPECT_EQ(p1_type_t::Size, 1);
    EXPECT_EQ(p1_type_t::Size, p.m_port_names.size());
    EXPECT_EQ(typeid(p1_type_t::nth_t<0>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(p.m_port_names[0], "an_int");

    EXPECT_EQ(p2_type_t::Size, 6);
    EXPECT_EQ(p2_type_t::Size, p2.m_port_names.size());
    EXPECT_EQ(typeid(p2_type_t::nth_t<0>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<1>).hash_code(), typeid(int).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<2>).hash_code(), typeid(double).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<3>).hash_code(), typeid(float).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<4>).hash_code(), typeid(std::string).hash_code());
    EXPECT_EQ(typeid(p2_type_t::nth_t<5>).hash_code(), typeid(std::stringstream).hash_code());
    EXPECT_EQ(p2.m_port_names[0], "an_int");
    EXPECT_EQ(p2.m_port_names[1], "another_int");
    EXPECT_EQ(p2.m_port_names[2], "a_double");
    EXPECT_EQ(p2.m_port_names[3], "a_float");
    EXPECT_EQ(p2.m_port_names[4], "a_string");
    EXPECT_EQ(p2.m_port_names[5], "a_sstream");
}

TEST_F(SegmentComponentTests, SegmentDefinitionCreation)
{
    auto segdef = segment::Definition("segdef_test", m_initializer);

    EXPECT_EQ(segdef.name(), "segdef_test");
}

TEST_F(SegmentComponentTests, SegmentDefinitionInstantiation)
{
    using p1_type_t = IngressPorts<int, size_t, uint64_t, unsigned short>;
    using p2_type_t = EgressPorts<int, int, double, float, std::string, std::string>;

    auto p1 = p1_type_t({"an_int", "a_sizet", "a_uint64t", "an_ushort"});
    auto p2 = p2_type_t({"an_int", "another_int", "a_double", "a_float", "a_string", "a_string2"});

    auto segdef = segment::Definition("segdef_test", m_initializer);

    segdef.attach_ingress_interface(p1);
    EXPECT_EQ(segdef.ingress_port_names().size(), p1.m_port_names.size());

    for (auto i = 0; i < p1_type_t::Size; i++)
    {
        EXPECT_EQ(segdef.ingress_port_names()[i], p1.m_port_names[i]);
    }

    segdef.attach_egress_interface(p2);
    EXPECT_EQ(segdef.egress_port_names().size(), p2.m_port_names.size());

    for (auto i = 0; i < p2_type_t::Size; i++)
    {
        EXPECT_EQ(segdef.egress_port_names()[i], p2.m_port_names[i]);
    }
}

TEST_F(SegmentComponentTests, SegmentNodeConstructionRxNode)
{
    using sink_type_t    = std::string;
    using source_type_t  = std::string;
    using rx_node_type_t = rx::RxNode<sink_type_t, source_type_t>;

    auto init = [this](segment::Builder& segment) {
        auto node_name = "xyz";
        auto node      = std::shared_ptr<rx_node_type_t>(rx::RxBuilder::Node<sink_type_t, source_type_t>::allocate(
            segment,
            node_name,
            rxcpp::operators::tap([](std::string s) { VLOG(10) << "Side Effect[Before]: " << s << std::endl; }),
            rxcpp::operators::map([](std::string s) { return s + "-Mapped"; }),
            rxcpp::operators::tap([](std::string s) { VLOG(10) << "Side Effect[After]: " << s << std::endl; })));

        auto seg_node = std::make_shared<SegmentNodeInfo<sink_type_t, source_type_t>>(segment, node);

        EXPECT_EQ(seg_node->name(), node_name);

        EXPECT_EQ(check_get_ingress(seg_node), false);
        EXPECT_EQ(check_get_egress(seg_node), false);
        EXPECT_EQ(check_register_port(seg_node), false);

        EXPECT_EQ(check_rx_node_upcast(seg_node), true);
        EXPECT_EQ(check_rx_source_upcast(seg_node), true);
        EXPECT_EQ(check_rx_sink_upcast(seg_node), true);
        EXPECT_EQ(check_source_upcast(seg_node), true);
        EXPECT_EQ(check_sink_upcast(seg_node), true);
        EXPECT_EQ(seg_node->is_running(), false);
    };

    auto segdef = segment::Definition::create("segment_test", init);
    auto seg    = Segment::instantiate(*segdef, m_resources);
}

TEST_F(SegmentComponentTests, SegmentNodeConstructionRxSource)
{
    using source_type_t    = std::string;
    using rx_source_type_t = rx::RxSource<source_type_t>;

    auto init = [this](segment::Builder& segment) {
        auto node_name   = "xyz";
        auto constructor = [&](rxcpp::subscriber<std::string> s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        };

        auto node = std::shared_ptr<rx_source_type_t>(
            rx::RxBuilder::Source<source_type_t>::allocate(segment, node_name, constructor));

        auto seg_node = std::make_shared<SegmentNodeInfo<source_type_t, source_type_t>>(segment, node);

        EXPECT_EQ(seg_node->name(), node_name);

        EXPECT_EQ(check_get_ingress(seg_node), false);
        EXPECT_EQ(check_get_egress(seg_node), false);
        EXPECT_EQ(check_register_port(seg_node), false);

        EXPECT_EQ(check_rx_node_upcast(seg_node), false);
        EXPECT_EQ(check_rx_source_upcast(seg_node), true);
        EXPECT_EQ(check_rx_sink_upcast(seg_node), false);
        EXPECT_EQ(check_source_upcast(seg_node), true);
        EXPECT_EQ(check_sink_upcast(seg_node), false);

        EXPECT_EQ(seg_node->is_running(), false);
    };

    auto segdef = segment::Definition::create("segment_test", init);
    auto seg    = Segment::instantiate(*segdef, m_resources);
}

TEST_F(SegmentComponentTests, SegmentNodeConstructionRxSink)
{
    using sink_type_t    = std::string;
    using rx_sink_type_t = rx::RxSink<sink_type_t>;

    auto init = [this](segment::Builder& segment) {
        auto node_name   = "xyz";
        auto constructor = [&](rxcpp::subscriber<std::string> s) {
            s.on_next("One");
            s.on_next("Two");
            s.on_next("Three");
            s.on_completed();
        };

        auto node = std::shared_ptr<rx_sink_type_t>(rx::RxBuilder::Sink<std::string>::allocate(
            segment,
            node_name,
            rxcpp::make_observer_dynamic<std::string>([](std::string x) { DVLOG(1) << x << std::endl; },
                                                      []() { DVLOG(1) << "Completed" << std::endl; })));

        auto seg_node = std::make_shared<SegmentNodeInfo<sink_type_t, sink_type_t>>(segment, node);

        EXPECT_EQ(seg_node->name(), node_name);

        EXPECT_EQ(check_get_ingress(seg_node), false);
        EXPECT_EQ(check_get_egress(seg_node), false);
        EXPECT_EQ(check_register_port(seg_node), false);

        EXPECT_EQ(check_rx_node_upcast(seg_node), false);
        EXPECT_EQ(check_rx_source_upcast(seg_node), false);
        EXPECT_EQ(check_rx_sink_upcast(seg_node), true);
        EXPECT_EQ(check_source_upcast(seg_node), false);
        EXPECT_EQ(check_sink_upcast(seg_node), true);
        EXPECT_EQ(seg_node->is_running(), false);
    };

    auto segdef = segment::Definition::create("segment_test", init);
    auto seg    = Segment::instantiate(*segdef, m_resources);
}

TEST_F(SegmentComponentTests, SegmentNodeConstructionIngress)
{
    using ingress_data_type_t = std::string;

    auto init = [this](segment::Builder& segment) {
        std::string node_name = "xyz";

        auto node     = std::make_shared<SegmentIngressPort<ingress_data_type_t>>(segment, node_name);
        auto seg_node = std::make_shared<SegmentNodeInfo<ingress_data_type_t, ingress_data_type_t>>(segment, node);

        EXPECT_EQ(seg_node->name(), node_name);

        EXPECT_EQ(check_get_ingress(seg_node), true);
        EXPECT_EQ(check_get_egress(seg_node), false);
        EXPECT_EQ(check_register_port(seg_node), true);

        EXPECT_EQ(check_rx_node_upcast(seg_node), false);
        EXPECT_EQ(check_rx_source_upcast(seg_node), false);
        EXPECT_EQ(check_rx_sink_upcast(seg_node), false);
        EXPECT_EQ(check_source_upcast(seg_node), true);
        EXPECT_EQ(check_sink_upcast(seg_node), false);
        EXPECT_EQ(seg_node->is_running(), false);
    };

    auto segdef = segment::Definition::create("segment_test", init);
    auto seg    = Segment::instantiate(*segdef, m_resources);
}

TEST_F(SegmentComponentTests, SegmentNodeConstructionEgress)
{
    using egress_data_type_t = std::string;

    auto init = [this](segment::Builder& segment) {
        auto node_name = "zyx";

        auto node     = std::make_shared<SegmentEgressPort<egress_data_type_t>>(segment, node_name);
        auto seg_node = std::make_shared<SegmentNodeInfo<egress_data_type_t, egress_data_type_t>>(segment, node);

        EXPECT_EQ(check_get_ingress(seg_node), false);
        EXPECT_EQ(check_get_egress(seg_node), true);
        EXPECT_EQ(check_register_port(seg_node), true);

        EXPECT_EQ(check_rx_node_upcast(seg_node), false);
        EXPECT_EQ(check_rx_source_upcast(seg_node), false);
        EXPECT_EQ(check_rx_sink_upcast(seg_node), false);
        EXPECT_EQ(check_source_upcast(seg_node), false);
        EXPECT_EQ(check_sink_upcast(seg_node), true);
        EXPECT_EQ(seg_node->is_running(), false);
    };

    auto segdef = segment::Definition::create("segment_test", init);
    auto seg    = Segment::instantiate(*segdef, m_resources);
}

TEST_F(SegmentComponentTests, SegmentNodeBaseInterfaceChecks)
{
    using sink_type_t    = std::string;
    using source_type_t  = std::string;
    using rx_node_type_t = rx::RxNode<sink_type_t, source_type_t>;

    auto init = [this](segment::Builder& segment) {
        auto node_name = "xyz";
        auto node      = std::shared_ptr<rx_node_type_t>(rx::RxBuilder::Node<sink_type_t, source_type_t>::allocate(
            segment,
            node_name,
            rxcpp::operators::tap([](std::string s) { VLOG(10) << "Side Effect[Before]: " << s << std::endl; }),
            rxcpp::operators::map([](std::string s) { return s + "-Mapped"; }),
            rxcpp::operators::tap([](std::string s) { VLOG(10) << "Side Effect[After]: " << s << std::endl; })));

        auto seg_node      = std::make_shared<SegmentNodeInfo<sink_type_t, source_type_t>>(segment, node);
        auto seg_node_base = std::static_pointer_cast<SegmentNodeInfoBase>(seg_node);

        // Downcast
        EXPECT_EQ(seg_node_base->get_node_type(), SegmentNodeInfoBase::NodeType::internal);
        EXPECT_EQ(seg_node_base->sink_type_hash(), typeid(sink_type_t).hash_code());
        EXPECT_EQ(seg_node_base->source_type_hash(), typeid(source_type_t).hash_code());

        auto seg_object = seg_node_base->get_segment_object();
        EXPECT_EQ(seg_object->name(), seg_node_base->name());
        EXPECT_EQ(seg_object->concurrency(), seg_node_base->concurrency());
        EXPECT_EQ(seg_object->is_running(), seg_node_base->is_running());
        EXPECT_EQ(std::addressof(seg_object->segment()), std::addressof(segment));
        EXPECT_EQ(std::addressof(seg_object->segment()), std::addressof(seg_node_base->segment()));

        // Bad upcast
        try
        {
            auto seg_node_test = seg_node_base->get_segment_node<sink_type_t, double>();
            FAIL() << "Expected std::invalid_argument";
        } catch (std::invalid_argument e)
        {
            // lots of words
        } catch (...)
        {
            FAIL() << "Expected std::invalid_argument";
        }

        // Good upcast
        auto seg_node_upcast = seg_node_base->get_segment_node<sink_type_t, source_type_t>();
        EXPECT_EQ(seg_node->source_type_hash(), seg_node_upcast->source_type_hash());
        EXPECT_EQ(seg_node->sink_type_hash(), seg_node_upcast->sink_type_hash());
        EXPECT_EQ(std::addressof(seg_object->segment()), std::addressof(seg_node_upcast->segment()));
        EXPECT_EQ(std::addressof(seg_node->segment()), std::addressof(seg_node_upcast->segment()));

        EXPECT_EQ(seg_node_upcast->name(), node_name);

        EXPECT_EQ(check_get_ingress(seg_node_upcast), false);
        EXPECT_EQ(check_get_egress(seg_node_upcast), false);
        EXPECT_EQ(check_register_port(seg_node_upcast), false);

        EXPECT_EQ(check_rx_node_upcast(seg_node_upcast), true);
        EXPECT_EQ(check_rx_source_upcast(seg_node_upcast), true);
        EXPECT_EQ(check_rx_sink_upcast(seg_node_upcast), true);
        EXPECT_EQ(check_source_upcast(seg_node_upcast), true);
        EXPECT_EQ(check_sink_upcast(seg_node_upcast), true);
        EXPECT_EQ(seg_node_upcast->is_running(), false);
    };

    auto segdef = m_pipeline->make_segment("segment_test", init);

    Executor exec;

    exec.register_pipeline(std::move(m_pipeline));

    exec.start();

    exec.join();
}
