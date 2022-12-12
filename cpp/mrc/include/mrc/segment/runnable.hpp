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

#include "mrc/runnable/launch_control.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/launchable.hpp"
#include "mrc/runnable/launcher.hpp"
#include "mrc/runnable/runnable.hpp"
#include "mrc/segment/context.hpp"
#include "mrc/segment/object.hpp"

#include <glog/logging.h>

#include <memory>
#include <ostream>
#include <utility>

namespace mrc::segment {

template <typename NodeT>
class Runnable : public Object<NodeT>, public runnable::Launchable
{
  public:
    template <typename... ArgsT>
    Runnable(std::string name, ArgsT&&... args) : m_node(std::make_unique<NodeT>(std::forward<ArgsT>(args)...))
    {
        // Set the name in the Object class
        this->set_name(std::move(name));
    }

    Runnable(std::string name, std::unique_ptr<NodeT> node) : m_node(std::move(node))
    {
        CHECK(m_node);

        // Set the name in the Object class
        this->set_name(std::move(name));
    }

  private:
    NodeT* get_object() const final;
    std::unique_ptr<runnable::Launcher> prepare_launcher(runnable::LaunchControl& launch_control) final;

    std::unique_ptr<NodeT> m_node;
};

template <typename NodeT>
NodeT* Runnable<NodeT>::get_object() const
{
    return m_node.get();
}

template <typename NodeT>
std::unique_ptr<runnable::Launcher> Runnable<NodeT>::prepare_launcher(runnable::LaunchControl& launch_control)
{
    if constexpr (std::is_base_of_v<runnable::Runnable, NodeT>)
    {
        DVLOG(10) << "Preparing launcher for " << this->type_name() << " in segment";
        return launch_control.prepare_launcher_with_wrapped_context<segment::Context>(
            this->launch_options(), std::move(m_node), this->name());
    }
    else
    {
        DVLOG(10) << this->type_name() << " is not a Runnable; no Launcher will be created";
        return nullptr;
    }
}

}  // namespace mrc::segment
