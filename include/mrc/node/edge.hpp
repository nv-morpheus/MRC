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
#include "mrc/channel/forward.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/node/forward.hpp"

#include <glog/logging.h>

#include <memory>
#include <utility>

namespace mrc::node {

/**
 * @brief An Edge is an Ingress adaptor. This base class provides the storage for actual Channel on which the write will
 * occur, while the Ingress interface maybe of of a different type.
 *
 * @tparam SourceT
 * @tparam SinkT
 */
template <typename SourceT, typename SinkT>
class EdgeBase : public channel::Ingress<SourceT>
{
  public:
    using source_t = SourceT;
    using sink_t   = SinkT;

    EdgeBase(std::shared_ptr<channel::Ingress<SinkT>> ingress) : m_ingress(std::move(ingress)) {}

  protected:
    inline channel::Ingress<SinkT>& ingress()
    {
        return *m_ingress;
    }

  private:
    std::shared_ptr<channel::Ingress<SinkT>> m_ingress;
};

/**
 * @brief Edge is a final implementation of EdgeBase.
 *
 * @note While Edge is final, template specialization can be used to create customized Edges from SourceT to SinkT.
 *
 * @tparam SourceT
 * @tparam SinkT
 */
template <typename SourceT, typename SinkT>
struct Edge<SourceT, SinkT, std::enable_if_t<std::is_convertible_v<SourceT, SinkT>>> final
  : public EdgeBase<SourceT, SinkT>
{
    using EdgeBase<SourceT, SinkT>::EdgeBase;

    inline channel::Status await_write(SourceT&& data) final
    {
        return this->ingress().await_write(std::move(data));
    }
};

}  // namespace mrc::node
