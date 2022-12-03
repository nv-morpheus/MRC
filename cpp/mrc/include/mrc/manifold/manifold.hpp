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

#include "mrc/manifold/interface.hpp"
#include "mrc/node/sink_properties.hpp"
#include "mrc/node/source_properties.hpp"
#include "mrc/pipeline/resources.hpp"
#include "mrc/types.hpp"

#include <string>

namespace mrc::manifold {

class Manifold : public Interface
{
  public:
    Manifold(PortName port_name, pipeline::Resources& resources);

    const PortName& port_name() const final;

  protected:
    pipeline::Resources& resources();

    const std::string& info() const
    {
        return m_info;
    }

  private:
    void add_input(const SegmentAddress& address, node::SourcePropertiesBase* input_source) final;

    void add_output(const SegmentAddress& address, node::SinkPropertiesBase* output_sink) final;

    virtual void do_add_input(const SegmentAddress& address, node::SourcePropertiesBase* input_source) = 0;
    virtual void do_add_output(const SegmentAddress& address, node::SinkPropertiesBase* output_sink)   = 0;

    PortName m_port_name;
    pipeline::Resources& m_resources;
    std::string m_info;
};

}  // namespace mrc::manifold
