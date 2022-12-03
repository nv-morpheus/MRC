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
#include "mrc/manifold/load_balancer.hpp"

#include <memory>

namespace mrc::manifold {

template <typename T>
struct Factory final
{
    static std::shared_ptr<Interface> make_manifold(PortName port_name, pipeline::Resources& resources)
    {
        return std::make_shared<LoadBalancer<T>>(std::move(port_name), resources);
    }
};

}  // namespace mrc::manifold
