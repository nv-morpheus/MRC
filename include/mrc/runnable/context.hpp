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

#include "mrc/runnable/types.hpp"

#include <glog/logging.h>

#include <cstddef>
#include <exception>
#include <sstream>
#include <string>

namespace mrc::runnable {

class Runner;

/**
 * @brief Provides identity and a means of synchronoization to an instance of a Runnable with respect to other instances
 * of the same Runnable.
 *
 * A unique Context is provided by the Launcher for each concurrent instance of Runnable. The Context provides
 * the rank() of the current instances, the number of instances via size() and a barrier() method to collectively
 * synchronize all instances.
 */
class Context
{
  public:
    Context() = delete;
    Context(std::size_t rank, std::size_t size);
    virtual ~Context() = default;

    EngineType execution_context() const;

    std::size_t rank() const;
    std::size_t size() const;

    void lock();
    void unlock();
    void barrier();
    void yield();

    const std::string& info() const;

    template <typename ContextT>
    ContextT& as()
    {
        auto up = dynamic_cast<ContextT*>(this);
        CHECK(up);
        return *up;
    }

    static Context& get_runtime_context();

    void set_exception(std::exception_ptr exception_ptr);

  protected:
    void init(const Runner& runner);
    bool status() const;
    void finish();
    virtual void init_info(std::stringstream& ss);

  private:
    std::size_t m_rank;
    std::size_t m_size;
    std::string m_info{"Uninitialized Context"};
    std::exception_ptr m_exception_ptr{nullptr};
    const Runner* m_runner{nullptr};

    virtual void do_lock()                          = 0;
    virtual void do_unlock()                        = 0;
    virtual void do_barrier()                       = 0;
    virtual void do_yield()                         = 0;
    virtual EngineType do_execution_context() const = 0;

    friend class Runner;
};

}  // namespace mrc::runnable
