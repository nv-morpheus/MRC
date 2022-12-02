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

#include "mrc/runnable/context.hpp"

#include <string>

namespace mrc::segment {

template <typename ContextT = runnable::Context>
class Context : public ContextT
{
  public:
    template <typename... ArgsT>
    Context(std::size_t rank, std::size_t size, std::string name, ArgsT&&... args) :
      ContextT(std::move(rank), std::move(size), std::forward<ArgsT>(args)...),
      m_name(std::move(name))
    {
        static_assert(std::is_base_of_v<runnable::Context, ContextT>, "ContextT must derive from Context");
        VLOG(10) << "Init with name: " << m_name;
    }

    const std::string& name() const;

  protected:
    void init_info(std::stringstream& ss) override
    {
        ss << m_name << "; ";
        ContextT::init_info(ss);
    }

  private:
    std::string m_name;
};

}  // namespace mrc::segment
