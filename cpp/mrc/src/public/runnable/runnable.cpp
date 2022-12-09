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

#include "mrc/runnable/runnable.hpp"

#include "mrc/runnable/context.hpp"

#include <atomic>
#include <sstream>
#include <string>
#include <typeinfo>

#ifndef NDEBUG
    #include <cxxabi.h>
#endif

namespace {
// This used to be in type_utils.hpp in the global namespace but
// is only used here
inline std::string demangle_type_str(const std::string& instr)
{
#ifndef NDEBUG
    int status;
    return abi::__cxa_demangle(instr.c_str(), nullptr, nullptr, &status);
#else
    return instr;
#endif
}
}  // namespace

namespace mrc::runnable {

Runnable::Runnable()  = default;
Runnable::~Runnable() = default;

Runnable::State Runnable::state() const
{
    return m_state;
}

std::string Runnable::info(const Context& ctx) const
{
    std::stringstream ss;
    ss << "[" << demangle_type_str(typeid(*this).name()) << "; " << ctx.info() << "]";
    return ss.str();
}

void Runnable::update_state(State new_state)
{
    if (m_state < new_state)
    {
        m_state = new_state;
        on_state_update(m_state);
    }
}

void Runnable::on_state_update(const State& /*unused*/) {}

}  // namespace mrc::runnable
