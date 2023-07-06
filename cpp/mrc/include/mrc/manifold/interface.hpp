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

#pragma once

#include "mrc/core/utils.hpp"
#include "mrc/edge/forward.hpp"
#include "mrc/types.hpp"

#include <atomic>
#include <memory>
#include <set>
#include <vector>

namespace mrc::manifold {

struct ManifoldPolicyInfoBase
{
    ManifoldPolicyInfoBase(SegmentAddress _address, bool _is_local, size_t _points) :
      address(_address),
      is_local(_is_local),
      points(_points)
    {}

    SegmentAddress address;

    // Whether or not this segment connection is local
    bool is_local;

    // Relative weighting of this address. Percentage is points / (total points of all segments)
    size_t points;
};

struct ManifoldPolicyInputInfo : public ManifoldPolicyInfoBase
{
    ManifoldPolicyInputInfo(SegmentAddress _address,
                            bool _is_local,
                            size_t _points,
                            edge::IWritableAcceptorBase* _edge) :
      ManifoldPolicyInfoBase(_address, _is_local, _points),
      edge(_edge)
    {}
    edge::IWritableAcceptorBase* edge;
};

struct ManifoldPolicyOutputInfo : public ManifoldPolicyInfoBase
{
    ManifoldPolicyOutputInfo(SegmentAddress _address,
                             bool _is_local,
                             size_t _points,
                             edge::IWritableProviderBase* _edge) :
      ManifoldPolicyInfoBase(_address, _is_local, _points),
      edge(_edge)
    {}

    edge::IWritableProviderBase* edge;
};

// struct ManifoldPolicy;

// class ManifoldInputPolicy
// {
//   public:
//     ManifoldInputPolicy(const ManifoldPolicy& _parent, std::vector<ManifoldPolicyInputInfo> _inputs) :
//       parent(_parent),
//       inputs(std::move(_inputs))
//     {}

//     const ManifoldPolicy& parent;
//     std::vector<ManifoldPolicyInputInfo> inputs;
// };

// class ManifoldOutputPolicy
// {
//   public:
//     ManifoldOutputPolicy(const ManifoldPolicy& _parent, std::vector<ManifoldPolicyOutputInfo> _outputs) :
//       parent(_parent),
//       outputs(std::move(_outputs))
//     {}

//     const ManifoldPolicy& parent;
//     std::vector<ManifoldPolicyOutputInfo> outputs;

//     virtual SegmentAddress get_next_tag() const = 0;
// };

class ManifoldPolicy
{
  public:
    ManifoldPolicy() = default;

    ManifoldPolicy(std::vector<ManifoldPolicyInputInfo> inputs,
                   std::map<SegmentAddress, ManifoldPolicyOutputInfo> outputs) :
      inputs(std::move(inputs)),
      outputs(std::move(outputs))
    {
        auto keys = extract_keys(this->outputs);

        m_output_addresses = std::vector<SegmentAddress>(keys.begin(), keys.end());
    }

    ManifoldPolicy(const ManifoldPolicy& other) :
      inputs(other.inputs),
      outputs(other.outputs),
      m_msg_counter(other.m_msg_counter.load()),
      m_output_addresses(other.m_output_addresses)
    {}

    ManifoldPolicy(ManifoldPolicy&& other) :
      inputs(other.inputs),
      outputs(other.outputs),
      m_msg_counter(0),
      m_output_addresses(std::move(other.m_output_addresses))
    {
        m_msg_counter = m_msg_counter.exchange(0);
    }

    ManifoldPolicy& operator=(const ManifoldPolicy& other)
    {
        if (this == &other)
        {
            return *this;
        }

        inputs             = other.inputs;
        outputs            = other.outputs;
        m_msg_counter      = other.m_msg_counter.load();
        m_output_addresses = other.m_output_addresses;

        return *this;
    }

    ManifoldPolicy& operator=(ManifoldPolicy&& other)
    {
        if (this == &other)
        {
            return *this;
        }

        inputs             = std::move(other.inputs);
        outputs            = std::move(other.outputs);
        m_msg_counter      = other.m_msg_counter.exchange(0);
        m_output_addresses = std::move(other.m_output_addresses);

        return *this;
    }

    std::vector<ManifoldPolicyInputInfo> inputs;
    std::map<SegmentAddress, ManifoldPolicyOutputInfo> outputs;

    SegmentAddress get_next_tag()
    {
        return m_output_addresses[m_msg_counter++ % m_output_addresses.size()];
    }

  private:
    std::atomic_size_t m_msg_counter{0};
    std::vector<SegmentAddress> m_output_addresses;
};

struct Interface
{
    virtual ~Interface() = default;

    virtual const PortName& port_name() const = 0;

    virtual void start() = 0;
    virtual void join()  = 0;

    virtual void add_local_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) = 0;
    virtual void add_local_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) = 0;

    virtual edge::IWritableProviderBase& get_input_sink() const = 0;

    virtual void update_policy(ManifoldPolicy&& policy) = 0;

    // virtual void add_remote_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) = 0;
    // virtual void add_remote_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) = 0;

    // updates are ordered
    // first, inputs are updated (upstream segments have not started emitting - this is safe)
    // then, upstream segments are started,
    // then, outputs are updated
    // this ensures downstream segments have started and are immediately capaable of handling data
    virtual void update_inputs()  = 0;
    virtual void update_outputs() = 0;
};

}  // namespace mrc::manifold
