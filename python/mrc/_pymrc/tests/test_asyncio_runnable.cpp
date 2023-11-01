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

#include "test_pymrc.hpp"

#include "pymrc/asyncio_runnable.hpp"
#include "pymrc/executor.hpp"
#include "pymrc/pipeline.hpp"
#include "pymrc/types.hpp"
#include "pymrc/utilities/object_wrappers.hpp"

#include "mrc/coroutines/async_generator.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/options/engine_groups.hpp"
#include "mrc/options/options.hpp"
#include "mrc/options/topology.hpp"
#include "mrc/runnable/types.hpp"
#include "mrc/segment/builder.hpp"
#include "mrc/segment/object.hpp"

#include <boost/fiber/policy.hpp>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rxcpp/rx.hpp>

#include <atomic>
#include <coroutine>
#include <memory>
#include <string>

namespace py    = pybind11;
namespace pymrc = mrc::pymrc;
using namespace std::string_literals;
using namespace py::literals;

class TestWithPythonInterpreter : public ::testing::Test
{
  public:
    virtual void interpreter_setup() = 0;

  protected:
    void SetUp() override;

    void TearDown() override;

  private:
    static bool m_initialized;
};

bool TestWithPythonInterpreter::m_initialized;

void TestWithPythonInterpreter::SetUp()
{
    if (!m_initialized)
    {
        m_initialized = true;
        pybind11::initialize_interpreter();
        interpreter_setup();
    }
}

void TestWithPythonInterpreter::TearDown() {}

class __attribute__((visibility("default"))) TestAsyncioRunnable : public TestWithPythonInterpreter
{
    void interpreter_setup() override
    {
        pybind11::module_::import("mrc.core.coro");
    }
};

class PythonCallbackAsyncioRunnable : public pymrc::AsyncioRunnable<int, int>
{
  public:
    PythonCallbackAsyncioRunnable(pymrc::PyObjectHolder operation) : m_operation(std::move(operation)) {}

    mrc::coroutines::AsyncGenerator<int> on_data(int&& value) override
    {
        py::gil_scoped_acquire acquire;

        auto coroutine = m_operation(py::cast(value));

        pymrc::PyObjectHolder result;
        {
            py::gil_scoped_release release;

            result = co_await pymrc::coro::PyTaskToCppAwaitable(std::move(coroutine));
        }

        auto result_casted = py::cast<int>(result);

        py::gil_scoped_release release;

        co_yield result_casted;
    };

  private:
    pymrc::PyObjectHolder m_operation;
};

TEST_F(TestAsyncioRunnable, UseAsyncioTasks)
{
    py::object globals = py::globals();
    py::exec(
        R"(
            async def fn(value):
                import asyncio
                await asyncio.sleep(0)
                return value * 2
        )",
        globals);

    pymrc::PyObjectHolder fn = static_cast<py::object>(globals["fn"]);

    ASSERT_FALSE(fn.is_none());

    std::atomic<unsigned int> counter = 0;
    pymrc::Pipeline p;

    auto init = [&counter, &fn](mrc::segment::IBuilder& seg) {
        auto src = seg.make_source<int>("src", [](rxcpp::subscriber<int>& s) {
            if (s.is_subscribed())
            {
                s.on_next(5);
                s.on_next(10);
            }

            s.on_completed();
        });

        auto internal = seg.construct_object<PythonCallbackAsyncioRunnable>("internal", fn);

        auto sink = seg.make_sink<int>("sink", [&counter](int x) {
            counter.fetch_add(x, std::memory_order_relaxed);
        });

        seg.make_edge(src, internal);
        seg.make_edge(internal, sink);
    };

    p.make_segment("seg1"s, init);
    p.make_segment("seg2"s, init);

    auto options = std::make_shared<mrc::Options>();
    options->topology().user_cpuset("0");
    // AsyncioRunnable only works with the Thread engine due to asyncio loops being thread-specific.
    options->engine_factories().set_default_engine_type(runnable::EngineType::Thread);

    pymrc::Executor exec{options};
    exec.register_pipeline(p);

    exec.start();
    exec.join();

    EXPECT_EQ(counter, 60);
}
