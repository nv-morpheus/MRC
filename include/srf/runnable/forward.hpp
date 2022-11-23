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

namespace mrc::runnable {

class Context;

template <typename ContextT = Context>
class RuntimeContext;

template <typename ContextT>
class FiberContext;

template <typename ContextT>
class ThreadContext;

class Engine;
class FiberEngine;
class ThreadEngine;

class Runnable;

template <typename ContextT = Context>
class RunnableWithContext;

template <typename ContextT = Context>
using FiberRunnable = RunnableWithContext<FiberContext<ContextT>>;  // NOLINT

template <typename ContextT = Context>
using ThreadRunnable = RunnableWithContext<ThreadContext<ContextT>>;  // NOLINT

class Runner;

class LaunchControl;
class Launcher;
class Launchable;

template <typename T>
class Payload;

}  // namespace mrc::runnable
