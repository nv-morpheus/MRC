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

// IWYU doesn't like these reletive paths
#include "../internal/architect/assignment_manager.hpp"
#include "../internal/utils/contains.hpp"
#include "../internal/utils/parse_config.hpp"

#include <srf/protos/architect.pb.h>
#include <srf/channel/status.hpp>
#include <srf/core/addresses.hpp>
#include <srf/core/utils.hpp>
#include <srf/pipeline/pipeline.hpp>
#include <srf/segment/builder.hpp>
#include <srf/segment/definition.hpp>
#include <srf/segment/egress_ports.hpp>
#include <srf/segment/ingress_ports.hpp>
#include <srf/segment/segment.hpp>
#include <srf/types.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// IWYU thinks we need exception for segment.make_source
// IWYU pragma: no_include <exception>
// IWYU thinks we need iterator for initializer lists
// IWYU pragma: no_include <iterator>

// IWYU thinks we need  <type_traits> for add_const when we iterate with:
// for const auto& ...
// and the iterator type being returned isn't actually const
// IWYU pragma: no_include <type_traits>

// IWYU thinks we need protobufs for google::protobuf::uint32
// IWYU pragma: no_include <google/protobuf/stubs/port.h>

using namespace srf;

class AssignmentTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        m_pipeline = pipeline::make_pipeline();

        auto segment_initializer = [](segment::Builder& seg) {};

        // ideally we make this a true source (seg_1) and true source (seg_4)
        auto seg_1 = Segment::create("seg_1", segment::EgressPorts<int>({"my_int2"}), segment_initializer);
        auto seg_2 = Segment::create("seg_2",
                                     segment::IngressPorts<int>({"my_int2"}),
                                     segment::EgressPorts<int>({"my_int3"}),
                                     segment_initializer);
        auto seg_3 = Segment::create("seg_3",
                                     segment::IngressPorts<int>({"my_int3"}),
                                     segment::EgressPorts<int>({"my_int4"}),
                                     segment_initializer);
        auto seg_4 = Segment::create("seg_4", segment::IngressPorts<int>({"my_int4"}), segment_initializer);

        m_pipeline->register_segment(seg_1);
        m_pipeline->register_segment(seg_2);
        m_pipeline->register_segment(seg_3);
        m_pipeline->register_segment(seg_4);

        m_manager = std::make_unique<AssignmentManager>();

        auto pipeline = m_pipeline->serialize();

        m_manager->add_pipeline(0, pipeline);
        m_manager->add_pipeline(42, pipeline);
    }

    void TearDown() override
    {
        m_manager.reset();
    }

    void balanced_config(std::string, std::uint32_t);

    std::unique_ptr<AssignmentManager> m_manager;
    std::unique_ptr<pipeline::Pipeline> m_pipeline;
};

TEST_F(AssignmentTest, LifeCycle) {}

TEST_F(AssignmentTest, ParseConfigDefault)
{
    auto maps = parse_config("*:1:*");

    for (const auto& map : maps)
    {
        auto [names, concurrency, groups] = map;

        EXPECT_EQ(names.size(), 1);
        EXPECT_TRUE(contains(names, "*"));

        EXPECT_EQ(concurrency, 1);

        EXPECT_EQ(groups.size(), 0);
    }
}

TEST_F(AssignmentTest, ParseConfig1)
{
    auto maps = parse_config("seg1,seg4:1:0;*:2:0,2-4,6");

    {
        auto [names, concurrency, groups] = maps[0];

        EXPECT_EQ(names.size(), 2);
        EXPECT_TRUE(contains(names, "seg1"));
        EXPECT_TRUE(contains(names, "seg4"));

        EXPECT_EQ(concurrency, 1);

        EXPECT_EQ(groups.size(), 1);
        EXPECT_EQ(groups[0], 0);
    }

    {
        auto [names, concurrency, groups] = maps[1];

        EXPECT_EQ(names.size(), 1);
        EXPECT_TRUE(contains(names, "*"));

        EXPECT_EQ(concurrency, 2);

        EXPECT_EQ(groups.size(), 5);
        std::vector<InstanceID> expected_groups = {0, 2, 3, 4, 6};
        EXPECT_EQ(groups, expected_groups);
    }
}

void AssignmentTest::balanced_config(std::string config_str, std::uint32_t concurrency)
{
    auto config_map                      = parse_config(config_str);
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {2, 3, 4};

    auto instance_to_configs = AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids);

    ASSERT_EQ(instance_to_configs.size(), 3);

    for (const auto& kv : instance_to_configs)
    {
        ASSERT_EQ(kv.second.segments_size(), 4);
        for (const auto& segment : kv.second.segments())
        {
            ASSERT_EQ(segment.concurrency(), concurrency);
        }
    }

    for (const auto& [instance_id, pipeline_config] : instance_to_configs)
    {
        m_manager->add_pipeline_config(0, pipeline_config);
    }

    auto assignments = m_manager->evaluate_state();

    ASSERT_EQ(m_pipeline->segment_count() * instance_ids.size() * concurrency, assignments.size());

    std::map<SegmentID, std::set<SegmentRank>> segment_ranks;

    for (const auto& [address, assignment] : assignments)
    {
        auto [id, rank] = segment_address_decode(address);
        ASSERT_FALSE(assignment.issue_event_on_complete());
        ASSERT_EQ(assignment.egress_polices_size(), m_pipeline->find_segment(id)->egress_port_names().size());
        for (const auto& [port_id, policy] : assignment.egress_polices())
        {
            ASSERT_EQ(policy.segment_addresses_size(), 1);
        }
        segment_ranks[id].insert(rank);
    }

    ASSERT_EQ(segment_ranks.size(), m_pipeline->segment_count());

    for (const auto& [id, ranks] : segment_ranks)
    {
        ASSERT_EQ(ranks.size(), instance_ids.size() * concurrency);
    }
}

TEST_F(AssignmentTest, GeneratePipelineConfig)
{
    balanced_config("*:1:*", 1);
}

TEST_F(AssignmentTest, GeneratePipelineConfigX2)
{
    balanced_config("*:2:*", 2);
}

TEST_F(AssignmentTest, GeneratePipelineConfigComplex)
{
    auto config_map                      = parse_config("seg_1,seg_4:1:0;seg_2:2:1;seg_3:1:1-2");
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {2, 3, 4};

    auto instance_to_configs = AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids);

    ASSERT_EQ(instance_ids.size(), 3);

    ASSERT_EQ(instance_to_configs[2].segments_size(), 2);
    ASSERT_EQ(instance_to_configs[3].segments_size(), 2);
    ASSERT_EQ(instance_to_configs[4].segments_size(), 1);

    for (const auto& [instance_id, pipeline_config] : instance_to_configs)
    {
        m_manager->add_pipeline_config(0, pipeline_config);
    }

    auto assignments = m_manager->evaluate_state();

    std::map<InstanceID, std::map<SegmentID, std::set<SegmentRank>>> segment_map;
    for (const auto& [segment_address, assignment] : assignments)
    {
        ASSERT_EQ(segment_address, assignment.address());
        auto [id, rank] = segment_address_decode(segment_address);
        segment_map[assignment.instance_id()][id].insert(rank);

        ASSERT_FALSE(assignment.issue_event_on_complete());
    }

    std::set<SegmentID> inst_2 = {hash("seg_1"), hash("seg_4")};
    std::set<SegmentID> inst_3 = {hash("seg_2"), hash("seg_3")};
    std::set<SegmentID> inst_4 = {hash("seg_3")};

    ASSERT_EQ(extract_keys(segment_map[2]), inst_2);
    ASSERT_EQ(extract_keys(segment_map[3]), inst_3);
    ASSERT_EQ(extract_keys(segment_map[4]), inst_4);

    ASSERT_EQ(segment_map[2][hash("seg_1")].size(), 1);
    ASSERT_EQ(segment_map[2][hash("seg_4")].size(), 1);
    ASSERT_EQ(segment_map[3][hash("seg_2")].size(), 2);
    ASSERT_EQ(segment_map[3][hash("seg_3")].size(), 1);
    ASSERT_EQ(segment_map[4][hash("seg_3")].size(), 1);

    ASSERT_EQ(assignments[segment_address_encode(hash("seg_1"), 0)]
                  .egress_polices()
                  .at(hash("my_int2"))
                  .segment_addresses_size(),
              2);

    // seg_2 and seg_3 each have 2 instances, so they are balanced, thus
    // we don't balance these, instead s2,r0 -> s3,r0 and s2,r1 -> s3, r1
    ASSERT_EQ(assignments[segment_address_encode(hash("seg_2"), 0)]
                  .egress_polices()
                  .at(hash("my_int3"))
                  .segment_addresses_size(),
              1);

    ASSERT_EQ(assignments[segment_address_encode(hash("seg_3"), 0)]
                  .egress_polices()
                  .at(hash("my_int4"))
                  .segment_addresses_size(),
              1);
}

TEST_F(AssignmentTest, GeneratePipelineConfigComplexMultiNode)
{
    auto pipeline = m_pipeline->serialize();

    // machine a (0)
    auto config_map_0                      = parse_config("seg_1,seg_4");
    std::vector<InstanceID> instance_ids_0 = {0};

    // machine b (42)
    auto config_map_42                      = parse_config("seg_2:2;seg_3:4:1");
    std::vector<InstanceID> instance_ids_42 = {1, 2};

    auto instance_to_configs_0  = AssignmentManager::make_pipeline_configs(config_map_0, pipeline, instance_ids_0);
    auto instance_to_configs_42 = AssignmentManager::make_pipeline_configs(config_map_42, pipeline, instance_ids_42);

    ASSERT_FALSE(m_manager->can_start());

    for (const auto& [instance_id, pipeline_config] : instance_to_configs_0)
    {
        m_manager->add_pipeline_config(0, pipeline_config);
    }

    ASSERT_FALSE(m_manager->can_start());

    for (const auto& [instance_id, pipeline_config] : instance_to_configs_42)
    {
        m_manager->add_pipeline_config(42, pipeline_config);
    }

    ASSERT_TRUE(m_manager->can_start());

    auto assignments = m_manager->evaluate_state();

    std::map<InstanceID, std::map<SegmentID, std::set<SegmentRank>>> segment_map;
    for (const auto& [segment_address, assignment] : assignments)
    {
        ASSERT_EQ(segment_address, assignment.address());
        auto [id, rank] = segment_address_decode(segment_address);
        segment_map[assignment.instance_id()][id].insert(rank);
    }

    std::set<SegmentID> inst_0 = {hash("seg_1"), hash("seg_4")};
    std::set<SegmentID> inst_1 = {hash("seg_2")};
    std::set<SegmentID> inst_2 = {hash("seg_2"), hash("seg_3")};

    ASSERT_EQ(extract_keys(segment_map[0]), inst_0);
    ASSERT_EQ(extract_keys(segment_map[1]), inst_1);
    ASSERT_EQ(extract_keys(segment_map[2]), inst_2);

    ASSERT_EQ(segment_map[0][hash("seg_1")].size(), 1);
    ASSERT_EQ(segment_map[0][hash("seg_4")].size(), 1);
    ASSERT_EQ(segment_map[1][hash("seg_2")].size(), 2);
    ASSERT_EQ(segment_map[2][hash("seg_2")].size(), 2);
    ASSERT_EQ(segment_map[2][hash("seg_3")].size(), 4);

    SegmentAddress seg_1_0_addr = segment_address_encode(hash("seg_1"), 0);
    SegmentAddress seg_2_0_addr = segment_address_encode(hash("seg_2"), 0);
    SegmentAddress seg_2_2_addr = segment_address_encode(hash("seg_2"), 2);
    SegmentAddress seg_3_0_addr = segment_address_encode(hash("seg_3"), 0);
    SegmentAddress seg_3_2_addr = segment_address_encode(hash("seg_3"), 2);
    SegmentAddress seg_4_0_addr = segment_address_encode(hash("seg_4"), 0);

    // check machine id
    ASSERT_EQ(assignments[seg_1_0_addr].machine_id(), 0);
    ASSERT_EQ(assignments[seg_4_0_addr].machine_id(), 0);
    ASSERT_EQ(assignments[seg_2_0_addr].machine_id(), 42);
    ASSERT_EQ(assignments[seg_3_0_addr].machine_id(), 42);
    ASSERT_EQ(assignments[seg_2_2_addr].machine_id(), 42);
    ASSERT_EQ(assignments[seg_3_2_addr].machine_id(), 42);

    // check issue event status
    ASSERT_EQ(assignments[seg_1_0_addr].issue_event_on_complete(), true);
    ASSERT_EQ(assignments[seg_4_0_addr].issue_event_on_complete(), false);
    ASSERT_EQ(assignments[seg_2_0_addr].issue_event_on_complete(), false);
    ASSERT_EQ(assignments[seg_3_0_addr].issue_event_on_complete(), true);
    ASSERT_EQ(assignments[seg_2_2_addr].issue_event_on_complete(), false);
    ASSERT_EQ(assignments[seg_3_2_addr].issue_event_on_complete(), true);

    // check network ingress connections
    ASSERT_EQ(assignments[seg_1_0_addr].network_ingress_ports_size(), 0);
    ASSERT_EQ(assignments[seg_4_0_addr].network_ingress_ports_size(), 1);
    ASSERT_EQ(assignments[seg_2_0_addr].network_ingress_ports_size(), 1);
    ASSERT_EQ(assignments[seg_3_0_addr].network_ingress_ports_size(), 0);
    ASSERT_EQ(assignments[seg_2_2_addr].network_ingress_ports_size(), 1);
    ASSERT_EQ(assignments[seg_3_2_addr].network_ingress_ports_size(), 0);

    ASSERT_EQ(assignments[seg_1_0_addr].egress_polices().at(hash("my_int2")).segment_addresses_size(), 4);

    // seg_2 and seg_3 each have 2 instances, so they are balanced, thus
    // we don't balance these, instead s2,r0 -> s3,r0 and s2,r1 -> s3, r1
    ASSERT_EQ(assignments[segment_address_encode(hash("seg_2"), 0)]
                  .egress_polices()
                  .at(hash("my_int3"))
                  .segment_addresses_size(),
              1);

    ASSERT_EQ(assignments[segment_address_encode(hash("seg_3"), 0)]
                  .egress_polices()
                  .at(hash("my_int4"))
                  .segment_addresses_size(),
              1);

    // issue an on complete for seg_1_0_addr
    m_manager->segment_on_complete(seg_1_0_addr);

    auto new_assignments = m_manager->evaluate_state();

    ASSERT_EQ(new_assignments[seg_2_0_addr].network_ingress_ports_size(), 0);
    ASSERT_EQ(new_assignments[seg_2_2_addr].network_ingress_ports_size(), 0);
}

TEST_F(AssignmentTest, GeneratePipelineConfigIncorrectInstanceIDs)
{
    // instance_ids must be continuous and monotonically increasing
    auto config_map                      = parse_config("*:1:*");
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {2, 4};

    EXPECT_DEATH(AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids),
                 "instance_ids must be continuous and increase monotonically");
}

TEST_F(AssignmentTest, GeneratePipelineConfigInstanceIDsMonotonicContinuousDecreasing)
{
    // instance_ids must be continuous and monotonically increasing
    auto config_map                      = parse_config("*:1:*");
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {4, 3, 2};

    EXPECT_DEATH(AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids),
                 "instance_ids must be continuous and increase monotonically");
}

TEST_F(AssignmentTest, GeneratePipelineConfigIncorrectGroupSize)
{
    // dies because the groups vec from the config map is order 3 {1,2,3} and the instance_ids is order 2 {2,3}.
    // groups.size() <= instance_ids.size()
    auto config_map                      = parse_config("*:1:1-3");
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {2, 3};

    EXPECT_DEATH(AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids),
                 "group size is larger than local instance count");
}

TEST_F(AssignmentTest, GeneratePipelineConfigIncorrectGroupValues)
{
    // dies because 3 does not index into the vector of size 2 with values { 2, 3 };
    auto config_map                      = parse_config("*:1:1,3");
    auto pipeline                        = m_pipeline->serialize();
    std::vector<InstanceID> instance_ids = {2, 3};

    EXPECT_DEATH(AssignmentManager::make_pipeline_configs(config_map, pipeline, instance_ids), "invalid group id");
}

TEST_F(AssignmentTest, AddConfig) {}
