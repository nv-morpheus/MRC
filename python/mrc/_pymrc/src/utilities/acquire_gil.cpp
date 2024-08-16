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

#include "pymrc/utilities/acquire_gil.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

namespace mrc::pymrc {

namespace py = pybind11;

AcquireGIL::AcquireGIL() : m_gil(std::make_unique<py::gil_scoped_acquire>()) {}

AcquireGIL::~AcquireGIL() = default;

inline void AcquireGIL::inc_ref()
{
    if (m_gil)
    {
        m_gil->inc_ref();
    }
}

inline void AcquireGIL::dec_ref()
{
    if (m_gil)
    {
        m_gil->dec_ref();
    }
}

void AcquireGIL::disarm()
{
    if (m_gil)
    {
        m_gil->disarm();
    }
}

void AcquireGIL::release()
{
    // Just delete the GIL object early
    m_gil.reset();
}

}  // namespace mrc::pymrc
