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
#include "mrc/pipeline/resources.hpp"

namespace mrc::manifold {

struct Connectable
{
    virtual ~Connectable() = default;

    /**
     * @brief Create a Manifold in the typed environment of the Connectable object, e.g. IngressPort, EgressPort
     * @return std::shared_ptr<manifold::Interface>
     */
    virtual std::shared_ptr<manifold::Interface> make_manifold(pipeline::Resources&) = 0;

    /**
     * @brief Connect a Connectable to a Manifold
     *
     * This method is called by the Connectable object, e.g. an IngressPort or EgressPort, which will form the proper
     * edge connections. The calling object determines if it's connecting to the ingress or the egress side of the
     * manifold.
     *
     * @param manifold
     */
    virtual void connect_to_manifold(std::shared_ptr<manifold::Interface> manifold) = 0;
};

}  // namespace mrc::manifold
