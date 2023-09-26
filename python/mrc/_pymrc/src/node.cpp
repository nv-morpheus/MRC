/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pymrc/node.hpp"

#include "pymrc/executor.hpp"

#include <pybind11/gil.h>

#include <memory>
#include <mutex>
#include <thread>

namespace mrc::pymrc {

PythonNodeLoopHandle::PythonNodeLoopHandle()
{
    pybind11::gil_scoped_acquire acquire;

    auto asyncio = pybind11::module_::import("asyncio");

    auto setup_debugging = create_gil_initializer();

    m_loop        = asyncio.attr("new_event_loop")();
    m_loop_ct     = false;
    m_loop_thread = std::thread([loop = m_loop, &ct = m_loop_ct, setup_debugging = std::move(setup_debugging)]() {
        setup_debugging();

        while (not ct)
        {
            {
                // run event loop once
                pybind11::gil_scoped_acquire acquire;
                loop.attr("stop")();
                loop.attr("run_forever")();
            }

            std::this_thread::yield();
        }

        pybind11::gil_scoped_acquire acquire;
        auto shutdown = loop.attr("shutdown_asyncgens")();
        loop.attr("run_until_complete")(shutdown);
        loop.attr("close")();
    });
}

PythonNodeLoopHandle::~PythonNodeLoopHandle()
{
    if (m_loop_thread.joinable())
    {
        m_loop_ct = true;
        m_loop_thread.join();
    }
}

uint32_t PythonNodeLoopHandle::inc_ref()
{
    return ++m_references;
}

uint32_t PythonNodeLoopHandle::dec_ref()
{
    return --m_references;
}

PyHolder PythonNodeLoopHandle::get_asyncio_event_loop()
{
    return m_loop;
}

PythonNodeContext::PythonNodeContext(const mrc::runnable::Runner& runner,
                                     mrc::runnable::IEngine& engine,
                                     std::size_t rank,
                                     std::size_t size) :
  mrc::runnable::Context(runner, engine, rank, size)
{
    if (m_loop_handle == nullptr)
    {
        m_loop_handle = std::make_unique<PythonNodeLoopHandle>();
    }

    m_loop_handle->inc_ref();
}

PythonNodeContext::~PythonNodeContext()
{
    if (m_loop_handle != nullptr and m_loop_handle->dec_ref() == 0)
    {
        m_loop_handle.reset();
    }
}

PyHolder PythonNodeContext::get_asyncio_event_loop()
{
    return m_loop_handle->get_asyncio_event_loop();
}

}  // namespace mrc::pymrc
