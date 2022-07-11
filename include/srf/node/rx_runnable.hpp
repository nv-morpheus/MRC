/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "srf/node/forward.hpp"
#include "srf/node/rx_execute.hpp"
#include "srf/node/rx_subscribable.hpp"
#include "srf/runnable/runnable.hpp"

namespace srf::node {

template <typename ContextT>
class RxRunnable : public runnable::RunnableWithContext<ContextT>, public RxSubscribable
{
    using state_t = runnable::Runnable::State;

  public:
    RxRunnable()           = default;
    ~RxRunnable() override = default;

  private:
    // implemented by node objects and will be final
    virtual void on_shutdown_critical_section()      = 0;
    virtual void on_stop(const rxcpp::subscription&) = 0;
    virtual void on_kill(const rxcpp::subscription&) = 0;

    // users can override these virtual method

    // the channel is still connected and data can still be emitted on the channel
    // but not from the subscriber object, but from the channel directly
    // exeucted by context with rank 0
    virtual void will_complete() {}

    // the last method that overridden and executed in the lifecycle of the runnable
    // any downstream channels will be closed at this point
    // exeucted by context with rank 0
    virtual void did_complete() {}

    void run(ContextT& ctx) final;
    void on_state_update(const state_t& state) final;

    void shutdown(ContextT& ctx);

    rxcpp::composite_subscription m_subscription;
};

template <typename ContextT>
void RxRunnable<ContextT>::run(ContextT& ctx)
{
    DVLOG(10) << ctx.info() << " creating composition subscription";
    rxcpp::composite_subscription subscription;
    m_subscription.add(subscription);
    ctx.barrier();
    DVLOG(10) << ctx.info() << " issuing subscribe";
    RxSubscribable::subscribe(subscription);
    DVLOG(10) << ctx.info() << " subscribe completed";
    shutdown(ctx);
    DVLOG(10) << ctx.info() << " shutdown completed";
}

template <typename ContextT>
void RxRunnable<ContextT>::shutdown(ContextT& ctx)
{
    ctx.barrier();
    if (ctx.rank() == 0)
    {
        DVLOG(10) << ctx.info() << " critical section shutdown - start";
        will_complete();
        on_shutdown_critical_section();
        did_complete();
        DVLOG(10) << ctx.info() << " critical section shutdown - finish";
    }
    ctx.barrier();
}

template <typename ContextT>
void RxRunnable<ContextT>::on_state_update(const state_t& state)
{
    switch (state)
    {
    case state_t::Stop:
        on_stop(m_subscription);
        break;

    case state_t::Kill:
        on_kill(m_subscription);
        break;

    default:
        break;
    }
}

}  // namespace srf::node
