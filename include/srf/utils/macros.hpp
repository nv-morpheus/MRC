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

#define SRF_CONCAT(prefix, suffix) prefix##suffix
#define SRF_CONCAT_EVAL(prefix, suffix) SRF_CONCAT(prefix, suffix)

// __COUNTER__ isnt standard but is supported by msvc, gcc and clang
#define SRF_UNIQUE_VAR_NAME(prefix) SRF_CONCAT_EVAL(prefix, __COUNTER__)

#ifndef DELETE_COPYABILITY
#define DELETE_COPYABILITY(foo) \
    foo(const foo&) = delete;   \
    foo& operator=(const foo& other) = delete;
#endif

#ifndef DELETE_MOVEABILITY
#define DELETE_MOVEABILITY(foo)   \
    foo(foo&&) noexcept = delete; \
    foo& operator=(foo&& other) noexcept = delete;
#endif

#ifndef DEFAULT_MOVEABILITY
#define DEFAULT_MOVEABILITY(foo)   \
    foo(foo&&) noexcept = default; \
    foo& operator=(foo&& other) noexcept = default;
#endif
