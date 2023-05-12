/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <memory>

namespace mrc {

#ifdef MRC_CODECOV_ENABLED
    #define SKIP_IF_CODE_COV() GTEST_SKIP() << "Skipping test when code coverage is enabled";
#else
    #define SKIP_IF_CODE_COV()
#endif

class Options;
}  // namespace mrc
namespace mrc::system {
class System;
}  // namespace mrc::system

std::shared_ptr<mrc::system::System> make_system(std::function<void(mrc::Options&)> updater = nullptr);
