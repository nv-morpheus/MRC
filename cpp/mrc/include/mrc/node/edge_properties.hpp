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

#pragma once

#include "mrc/channel/channel.hpp"
#include "mrc/channel/ingress.hpp"

#include <memory>

namespace mrc::node {

// If:
// - SourceChannel<T> is an IngressAcceptor
// - SinkChannel<T> can be either an IngressProvider or a ChannelAcceptor
// - Queue<T> is an IngressProivder and a ChannelProvider

// Then, Edges can be from:
// - make_edge(IngressAcceptor<T>, IngressProivder<T>)
// - make_edge(ChannelProvider<T>, ChannelAcceptor<T>)

// Queue<T> is not an IngressAcceptor<T> nor ChannelAcceptor<T>, therefore, edges cannot be formed between Queues
// - this is a good thing, a Queue is simply a Channel holder with no progress engine, thus unable to driver forward
// progress.

class EdgeBuilder;

template <typename T>
class ChannelProvider
{
  public:
    virtual ~ChannelProvider() = default;

  private:
    virtual std::shared_ptr<channel::Channel<T>> channel() = 0;
    friend EdgeBuilder;
};

template <typename T>
class ChannelAcceptor
{
  public:
    virtual ~ChannelAcceptor() = default;

  private:
    virtual void set_channel(std::shared_ptr<channel::Channel<T>> channel) = 0;
    friend EdgeBuilder;
};

}  // namespace mrc::node
