/**
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

#include "mrc/utils/string_utils.hpp"

#include <nlohmann/json_fwd.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>

namespace pybind11 {
class gil_scoped_acquire;
}  // namespace pybind11

namespace mrc::pymrc {

// Export everything in the mrc::pymrc namespace by default since we compile with -fvisibility=hidden
#pragma GCC visibility push(default)

/**
 * @brief Wraps a `pybind11::gil_scoped_acquire` with additional functionality to release the GIL before this object
 * leaves the scope. Useful to avoid unnecessary nested `gil_scoped_acquire` then `gil_scoped_release` which need to
 * grab the GIL twice
 *
 */
class AcquireGIL
{
  public:
    //   Create the object in place
    AcquireGIL();
    ~AcquireGIL();

    void inc_ref();

    void dec_ref();

    void disarm();

    /**
     * @brief Releases the GIL early. The GIL will only be released once.
     *
     */
    void release();

  private:
    // Use an unique_ptr here to allow releasing the GIL early
    std::unique_ptr<pybind11::gil_scoped_acquire> m_gil;
};

#pragma GCC visibility pop

}  // namespace mrc::pymrc
