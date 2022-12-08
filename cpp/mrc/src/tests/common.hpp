/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>

namespace mrc {

inline void skip_if_in_ci()
{
    if (std::getenv("CI") != nullptr)
    {
        GTEST_SKIP() << "Test skipped in CI";
    }
}

class Options;
}  // namespace mrc
namespace mrc::internal::system {
class System;
}  // namespace mrc::internal::system

std::shared_ptr<mrc::internal::system::System> make_system(std::function<void(mrc::Options&)> updater = nullptr);
