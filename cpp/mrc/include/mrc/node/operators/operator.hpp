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

#include "mrc/channel/ingress.hpp"
#include "mrc/node/sink_properties.hpp"

namespace mrc::node {

struct OperatorBase
{
    virtual ~OperatorBase() = 0;
};

inline OperatorBase::~OperatorBase() = default;

template <typename T>
class Operator : public SinkProperties<T>, public OperatorBase, public std::enable_shared_from_this<Operator<T>>
{
    // SinkProperties
    std::shared_ptr<channel::Ingress<T>> channel_ingress() final;

    // forwarding method
    virtual channel::Status on_next(T&& data) = 0;

    // called by the IngressAdaptor's destructor
    // this signifies that the last held IngressAdaptor has been releases
    // and the Operator should cascade the on_complete signal
    virtual void on_complete() = 0;

    // Ingress Adaptor
    class IngressAdaptor : public channel::Ingress<T>
    {
      public:
        IngressAdaptor(Operator& parent) : m_parent(parent) {}
        ~IngressAdaptor()
        {
            m_parent.on_complete();
        }

        inline channel::Status await_write(T&& data) final
        {
            return m_parent.on_next(std::move(data));
        }

      private:
        Operator& m_parent;
    };

    std::weak_ptr<IngressAdaptor> m_ingress;
    friend IngressAdaptor;
};

template <typename T>
std::shared_ptr<channel::Ingress<T>> Operator<T>::channel_ingress()
{
    std::shared_ptr<IngressAdaptor> ingress;
    if ((ingress = m_ingress.lock()))
    {
        return ingress;
    }
    auto this_operator = this->shared_from_this();
    ingress = std::shared_ptr<IngressAdaptor>(new IngressAdaptor(*this), [this_operator](IngressAdaptor* ptr) mutable {
        delete ptr;
        this_operator.reset();
    });
    auto m_ingress = ingress;
    return ingress;
}

}  // namespace mrc::node
