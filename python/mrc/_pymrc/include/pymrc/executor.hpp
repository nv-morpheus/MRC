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
#pragma once

#include "mrc/pipeline/executor.hpp"  // for IExecutor, State
#include "mrc/types.hpp"              // for Future, SharedFuture

#include <pybind11/pytypes.h>

#include <functional>  // for function
#include <future>      // for future & promise
#include <memory>

namespace mrc {
class Options;
}  // namespace mrc

namespace mrc::pymrc {
class Pipeline;

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

// This object serves as a bridge between awaiting in python and awaiting on fibers
class Awaitable : public std::enable_shared_from_this<Awaitable>
{
  public:
    Awaitable();

    Awaitable(Future<pybind11::object>&& _future);

    std::shared_ptr<Awaitable> iter();

    std::shared_ptr<Awaitable> await();

    void next();

  private:
    Future<pybind11::object> m_future;
};

class Executor
{
  public:
    Executor();

    // TODO: move callback to options?
    Executor(std::shared_ptr<Options> options, std::function<void(State)> state_change_cb = nullptr);
    ~Executor();

    void register_pipeline(pymrc::Pipeline& pipeline);

    void start();
    void stop();
    void join();
    std::shared_ptr<Awaitable> join_async();

    std::shared_ptr<pipeline::IExecutor> get_executor() const;

  private:
    SharedFuture<void> m_join_future;

    void change_stage(State new_state);

    std::shared_ptr<pipeline::IExecutor> m_exec;
    State m_state                                = State::Init;
    std::function<void(State)> m_state_change_cb = nullptr;
};

class PyBoostFuture
{
  public:
    PyBoostFuture();

    pybind11::object result();
    pybind11::object py_result();

    void set_result(pybind11::object&& obj);

  private:
    std::promise<pybind11::object> m_promise{};
    std::future<pybind11::object> m_future{};
};

#pragma GCC visibility pop

}  // namespace mrc::pymrc
