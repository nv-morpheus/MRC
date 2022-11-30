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

#include "mrc/runnable/context.hpp"

#include "mrc/runnable/runner.hpp"
#include "mrc/runnable/types.hpp"

#include <boost/fiber/fss.hpp>
#include <glog/logging.h>

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>
#include <utility>

namespace mrc::runnable {

namespace {

struct FiberLocalContext
{
    using fiber_local_t = boost::fibers::fiber_specific_ptr<FiberLocalContext>;

    static fiber_local_t& get()
    {
        static fiber_local_t fiber_local;
        return fiber_local;
    }

    Context* m_context{nullptr};
};

}  // namespace

Context::Context(std::size_t rank, std::size_t size) : m_rank(rank), m_size(size) {}

EngineType Context::execution_context() const
{
    return do_execution_context();
}

std::size_t Context::rank() const
{
    return m_rank;
}

std::size_t Context::size() const
{
    return m_size;
}

void Context::lock()
{
    if (m_size > 1)
    {
        do_lock();
    }
}

void Context::unlock()
{
    if (m_size > 1)
    {
        do_unlock();
    }
}

void Context::barrier()
{
    if (m_size > 1)
    {
        do_barrier();
    }
}

void Context::yield()
{
    do_yield();
}

void Context::init(const Runner& runner)
{
    auto& fiber_local = FiberLocalContext::get();
    fiber_local.reset(new FiberLocalContext());
    fiber_local->m_context = this;

    std::stringstream ss;
    this->init_info(ss);
    m_info = ss.str();

    m_runner = &runner;
}

void Context::finish()
{
    if (m_exception_ptr)
    {
        std::rethrow_exception(m_exception_ptr);
    }
}

void Context::set_exception(std::exception_ptr exception_ptr)
{
    try
    {
        std::rethrow_exception(exception_ptr);
    } catch (const std::exception& e)
    {
        LOG(ERROR) << info() << ": set_exception issued; issuing kill to current runnable. Exception msg: " << e.what();

        // Only save and call kill on first invocation
        if (m_exception_ptr == nullptr)
        {
            m_exception_ptr = std::move(std::current_exception());
            m_runner->kill();
        }
    }
}

Context& Context::get_runtime_context()
{
    auto& fiber_local = FiberLocalContext::get();
    CHECK(fiber_local.get()) << "context not set on this fiber";
    DCHECK(fiber_local->m_context) << "context not set on this fiber";
    return *fiber_local->m_context;
}

void Context::init_info(std::stringstream& ss)
{
    ss << "rank: " << rank() << "; size: " << size();
}

const std::string& Context::info() const
{
    return m_info;
}

bool Context::status() const
{
    return (m_exception_ptr == nullptr);
}

}  // namespace mrc::runnable
