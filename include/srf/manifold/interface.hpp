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

#include "srf/node/forward.hpp"
#include "srf/types.hpp"

namespace srf::manifold {

struct Interface
{
    virtual ~Interface()                                                                            = default;
    virtual const PortName& port_name() const                                                       = 0;
    virtual void start()                                                                            = 0;
    virtual void join()                                                                             = 0;
    virtual void add_input(const SegmentAddress& address, node::SourcePropertiesBase* input_source) = 0;
    virtual void add_output(const SegmentAddress& address, node::SinkPropertiesBase* output_sink)   = 0;

    // updates are ordered
    // first, inputs are updated (upstream segments have not started emitting - this is safe)
    // then, upstream segments are started,
    // then, outputs are updated
    // this ensures downstream segments have started and are immediately capaable of handling data
    virtual void update_inputs()  = 0;
    virtual void update_outputs() = 0;
};

}  // namespace srf::manifold
