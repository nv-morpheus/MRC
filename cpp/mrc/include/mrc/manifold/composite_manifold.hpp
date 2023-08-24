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

#include "mrc/channel/buffered_channel.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/channel/types.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/manifold/egress.hpp"
#include "mrc/manifold/ingress.hpp"
#include "mrc/manifold/manifold.hpp"
#include "mrc/node/operators/router.hpp"
#include "mrc/node/sink_channel_owner.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/runnable/context.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/segment/utils.hpp"
#include "mrc/types.hpp"

#include <boost/fiber/future/future.hpp>
#include <boost/fiber/operations.hpp>

#include <chrono>

namespace mrc::manifold {

template <typename T>
class ManifoldNode : public node::RouterReadableAcceptor<T, SegmentAddress>,
                     public node::RouterWritableAcceptor<T, SegmentAddress>,
                     public runnable::RunnableWithContext<runnable::Context>
{
  public:
    ManifoldNode() {}

    void add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source)
    {
        boost::fibers::packaged_task<void()> update_task([this, address, input_source] {
            mrc::make_edge(*input_source, *this->get_sink(address));
        });

        auto update_future = update_task.get_future();

        CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

        // Before continuing, wait for the update to be processed
        update_future.get();
    }

    void add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink)
    {
        boost::fibers::packaged_task<void()> update_task([this, address, output_sink] {
            mrc::make_edge(*this->get_source(address), *output_sink);
        });

        auto update_future = update_task.get_future();

        CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

        // Before continuing, wait for the update to be processed
        update_future.get();
    }

    void remove_input(const SegmentAddress& address)
    {
        boost::fibers::packaged_task<void()> update_task([this, address] {
            this->drop_source(address);
        });

        auto update_future = update_task.get_future();

        CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

        // Before continuing, wait for the update to be processed
        update_future.get();
    }

    void remove_output(const SegmentAddress& address)
    {
        boost::fibers::packaged_task<void()> update_task([this, address] {
            this->drop_sink(address);
        });

        auto update_future = update_task.get_future();

        CHECK_EQ(m_updates.await_write(std::move(update_task)), channel::Status::success);

        // Before continuing, wait for the update to be processed
        update_future.get();
    }

  private:
    void run(runnable::Context& ctx) final
    {
        std::uint64_t backoff = 128;
        T data;

        while (m_is_running)
        {
            // if we are rank 0, check for updates
            if (ctx.rank() == 0)
            {
                channel::Status update_status;
                boost::fibers::packaged_task<void()> next_update;

                while ((update_status = m_updates.try_read(next_update)) == channel::Status::success)
                {
                    // Run the next update
                    next_update();
                }
            }

            // Barrier to sync threads
            ctx.barrier();

            // Now pull from the queue. Dont wait for any time if there isnt a message
            auto status = this->get_readable_edge()->await_read_for(data, channel::duration_t::zero());

            if (status == channel::Status::success)
            {
                backoff = 1;
                this->get_writable_edge()->await_write(std::move(data));
            }
            else if (status == channel::Status::timeout)
            {
                // If there are no pending updates, sleep
                if (backoff < 1024)
                {
                    backoff = (backoff << 1);
                }
                boost::this_fiber::sleep_for(std::chrono::microseconds(backoff));
            }
            else
            {
                // Should not happen
                throw exceptions::MrcRuntimeError("Unexpected channel status in manifold: " << status);
            }
        }
    }

    bool m_is_running{true};
    channel::BufferedChannel<boost::fibers::packaged_task<void()>> m_updates;
};

template <typename IngressT, typename EgressT>
class CompositeManifold : public Manifold
{
    static_assert(std::is_base_of_v<IngressDelegate, IngressT>, "ingress must be derived from IngressDelegate");
    static_assert(std::is_base_of_v<EgressDelegate, EgressT>, "ingress must be derived from EgressDelegate");

  public:
    CompositeManifold(PortName port_name,
                      runnable::IRunnableResources& resources,
                      std::unique_ptr<IngressT> ingress,
                      std::unique_ptr<EgressT> egress) :
      Manifold(std::move(port_name), resources),
      m_ingress(std::move(ingress)),
      m_egress(std::move(egress))
    {
        // Already created, link them together
        mrc::make_edge(*m_ingress, *m_egress);
    }

    CompositeManifold(PortName port_name, runnable::IRunnableResources& resources) :
      CompositeManifold(std::move(port_name), resources, std::make_unique<IngressT>(), std::make_unique<EgressT>())
    {
        // Then link them together
        // mrc::make_edge(*m_ingress, *m_egress);

        // // construct IngressT and EgressT on the NUMA node / memory domain in which the object will run
        // this->resources()
        //     .main()
        //     .enqueue([this] {

        //     })
        //     .get();
    }

  protected:
    IngressT& ingress()
    {
        CHECK(m_ingress);
        return *m_ingress;
    }

    EgressT& egress()
    {
        CHECK(m_egress);
        return *m_egress;
    }

  private:
    void do_add_input(const SegmentAddress& address, edge::IWritableAcceptorBase* input_source) final
    {
        // enqueue update to be done later
        m_input_updates.push_back([this, address, input_source] {
            DVLOG(10) << info() << ": ingress attaching to upstream segment " << segment::info(address);
            m_ingress->add_input(address, input_source);
            on_add_input(address);
        });
    }

    void do_add_output(const SegmentAddress& address, edge::IWritableProviderBase* output_sink) final
    {
        // enqueue update to be done later
        m_output_updates.push_back([this, address, output_sink] {
            DVLOG(10) << info() << ": egress attaching to downstream segment " << segment::info(address);
            m_egress->add_output(address, output_sink);
            on_add_output(address);
        });
    }

    void update(std::vector<std::function<void()>>& updates)
    {
        // resources()
        //     .main()
        //     .enqueue([&] {
        //         for (auto& update_fn : updates)
        //         {
        //             update_fn();
        //         }
        //     })
        //     .get();
        updates.clear();
    }

    void update_inputs() final
    {
        will_update_inputs();
        if (!m_input_updates.empty())
        {
            DVLOG(10) << info() << ": issuing all enqueued input updates";
            update(m_input_updates);
            DVLOG(10) << port_name() << " manifold finished input updates";
        }
    }

    void update_outputs() final
    {
        will_update_outputs();
        if (!m_output_updates.empty())
        {
            DVLOG(10) << info() << ": issuing all enqueued output updates";
            update(m_output_updates);
            DVLOG(10) << port_name() << " manifold finished output updates";
        }
    }

    virtual void on_add_input(const SegmentAddress& address) {}
    virtual void on_add_output(const SegmentAddress& address) {}

    virtual void will_update_inputs() {}
    virtual void will_update_outputs() {}

    std::vector<std::function<void()>> m_input_updates;
    std::vector<std::function<void()>> m_output_updates;

    std::unique_ptr<IngressT> m_ingress;
    std::unique_ptr<EgressT> m_egress;
};

}  // namespace mrc::manifold
