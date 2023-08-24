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

#include "mrc/edge/edge_writable.hpp"
#include "mrc/manifold/interface.hpp"
#include "mrc/runnable/runnable_resources.hpp"
#include "mrc/types.hpp"

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>

namespace mrc::manifold {

// class Manifold : public Interface
// {
//   public:
//     Manifold(PortName port_name, runnable::IRunnableResources& resources);
//     ~Manifold() override;

//     const PortName& port_name() const final;

//   protected:
//     runnable::IRunnableResources& resources();

//     const std::string& info() const;

//   private:
//     void add_local_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) final;
//     void add_local_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) final;

//     void update_policy(ManifoldPolicy policy) override {}

//     virtual void do_add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) = 0;
//     virtual void do_add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) = 0;

//     PortName m_port_name;
//     runnable::IRunnableResources& m_resources;
//     std::string m_info;
// };

// class ManifoldNodeBase : public virtual edge::IWritableProviderBase,
//                          public virtual edge::IMultiWritableAcceptorBase<SegmentAddress>,
//                          public runnable::RunnableWithContext<runnable::Context>
// {
//   public:
//     virtual void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source);

//     virtual void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink);

//   protected:
//     ManifoldPolicy& current_policy();
//     const ManifoldPolicy& current_policy() const;

//     virtual edge::IWritableAcceptorBase& get_output(SegmentAddress address) const = 0;
//     virtual void drop_outputs()                                                   = 0;

//   private:
//     void run(runnable::Context& ctx) final;

//     void update_policy(ManifoldPolicy policy);

//     virtual void do_update_policy(const ManifoldPolicy& policy);

//     virtual channel::Status process_one() = 0;

//     bool m_is_running{true};
//     channel::BufferedChannel<mrc::PackagedTask<void()>> m_updates;
//     ManifoldPolicy m_current_policy;

//     friend class ManifoldBase;
// };

// // Utility class to avoid tagger and untagger getting mixed up
// class ManifoldTaggerBase : public ManifoldNodeBase
// {
//   public:
//     void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) override;

//   protected:
//     SegmentAddress get_next_tag();

//   private:
//     void do_update_policy(const ManifoldPolicy& policy) override;

//     std::atomic_size_t m_msg_counter{0};
// };

class ManifoldTaggerBase2 : public virtual edge::IWritableProviderBase,
                            public virtual edge::IMultiWritableAcceptorBase<InstanceID>
{
  public:
  protected:
    // Mutex used to protect the output from being updated while in use
    std::shared_mutex m_output_mutex;

    InstanceID get_next_tag()
    {
        return m_current_policy.get_next_tag();
    }

  private:
    void update_policy(ManifoldPolicy&& policy);

    virtual void add_output(InstanceID port_address, bool is_local, edge::IWritableProviderBase* output_sink) = 0;

    std::atomic_size_t m_msg_counter{0};
    ManifoldPolicy m_current_policy;

    friend class ManifoldBase;
};

// // Utility class to avoid tagger and untagger getting mixed up
// class ManifoldUnTaggerBase : public ManifoldNodeBase
// {};

// template <typename T>
// class ManifoldTagger : public ManifoldTaggerBase,
//                        public edge::IWritableProvider<T>,
//                        public node::SinkChannelOwner<T>,
//                        public node::RouterWritableAcceptor<SegmentAddress, std::pair<SegmentAddress, T>>
// {
//     edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
//     {
//         return *this->get_source(address);
//     }
//     channel::Status process_one() override
//     {
//         T data;

//         auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

//         if (status == channel::Status::success)
//         {
//             auto tag = this->get_next_tag();

//             this->get_writable_edge(tag)->await_write(std::make_pair(tag, std::move(data)));
//         }

//         return status;
//     }
// };

// template <typename T>
// class ManifoldUnTagger : public ManifoldUnTaggerBase,
//                          public edge::IWritableProvider<std::pair<SegmentAddress, T>>,
//                          public node::SinkChannelOwner<std::pair<SegmentAddress, T>>,
//                          public node::RouterWritableAcceptor<SegmentAddress, T>
// {
//     edge::IWritableAcceptorBase& get_output(SegmentAddress address) const override
//     {
//         return *this->get_source(address);
//     }

//     channel::Status process_one() override
//     {
//         std::pair<SegmentAddress, T> data;

//         auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

//         if (status == channel::Status::success)
//         {
//             // Use the tag to determine where it should go
//             auto tag = data.first;

//             this->get_writable_edge(tag)->await_write(std::move(data.second));
//         }

//         return status;
//     }
// };

class ManifoldBase : public Interface, public runnable::RunnableResourcesProvider
{
  public:
    ManifoldBase(runnable::IRunnableResources& resources,
                 std::string port_name,
                 std::unique_ptr<ManifoldTaggerBase2> tagger);

    const PortName& port_name() const override;

    void start() override;

    void join() override;

  protected:
    const std::string& info() const;

  private:
    void add_local_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) final;

    void add_local_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) final;

    edge::IWritableProviderBase& get_input_sink() const override;

    void update_policy(ManifoldPolicy&& policy) override;

    void update_inputs() override;
    void update_outputs() override;

    PortName m_port_name;
    // runnable::IRunnableResources& m_resources;
    std::string m_info;

    std::unique_ptr<ManifoldTaggerBase2> m_router_node;
};

// template <typename T>
// class TypedManifold : public ManifoldBase
// {
//     TypedManifold(std::string port_name) :
//       ManifoldBase(std::move(port_name), std::make_shared<ManifoldTagger<T>>(),
//       std::make_shared<ManifoldUnTagger<T>>())
//     {}
// };

}  // namespace mrc::manifold
