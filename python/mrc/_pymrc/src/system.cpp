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

#include "pymrc/system.hpp"

#include "mrc/engine/system/isystem.hpp"
#include "mrc/options/options.hpp"

#include <glog/logging.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <array>
#include <memory>
#include <ostream>
#include <utility>

namespace mrc::pymrc {

System::System(std::shared_ptr<Options> options) : internal::system::ISystem(std::move(options)) {}

SystemResources::SystemResources(std::shared_ptr<System> system) : internal::system::IResources(std::move(system))
{
    add_gil_initializer();
    add_gil_finalizer();
}

void SystemResources::add_gil_initializer()
{
    bool has_pydevd_trace = false;

    // We check if there is a debugger by looking at sys.gettrace() and seeing if the function contains 'pydevd'
    // somewhere in the module name. Its important to get this right because calling `debugpy.debug_this_thread()`
    // will fail if there is no debugger and can dramatically alter performanc
    auto sys = pybind11::module_::import("sys");

    auto trace_func = sys.attr("gettrace")();

    if (!trace_func.is_none())
    {
        auto trace_module = pybind11::getattr(trace_func, "__module__", pybind11::none());

        if (!trace_module.is_none() && !trace_module.attr("find")("pydevd").equal(pybind11::int_(-1)))
        {
            VLOG(10) << "Found pydevd trace function. Will attempt to enable debugging for MRC threads.";
            has_pydevd_trace = true;
        }
    }

    // Release the GIL for the remainder
    pybind11::gil_scoped_release nogil;

    internal::system::IResources::add_thread_initializer([has_pydevd_trace] {
        pybind11::gil_scoped_acquire gil;

        // Increment the ref once to prevent creating and destroying the thread state constantly
        gil.inc_ref();

        try
        {
            // Try to load debugpy only if we found a trace function
            if (has_pydevd_trace)
            {
                auto debugpy = pybind11::module_::import("debugpy");

                auto debug_this_thread = debugpy.attr("debug_this_thread");

                debug_this_thread();

                VLOG(10) << "Debugging enabled from mrc threads";
            }
        } catch (pybind11::error_already_set& err)
        {
            if (err.matches(PyExc_ImportError))
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Could import error on debugpy";
                // Fail silently
            }
            else
            {
                VLOG(10) << "Debugging disabled. Breakpoints will not be hit. Unknown error: " << err.what();
                // Rethrow everything else
                throw;
            }
        }
    });
}

void SystemResources::add_gil_finalizer()
{
    bool python_finalizing = _Py_IsFinalizing() != 0;

    if (python_finalizing)
    {
        // If python if finalizing, dont worry about thread state
        return;
    }

    // Ensure we dont have the GIL here otherwise this deadlocks.

    internal::system::IResources::add_thread_finalizer([] {
        bool python_finalizing = _Py_IsFinalizing() != 0;

        if (python_finalizing)
        {
            // If python if finalizing, dont worry about thread state
            return;
        }

        pybind11::gil_scoped_acquire gil;

        // Decrement the ref to destroy the GIL states
        gil.dec_ref();
    });
}

}  // namespace mrc::pymrc
