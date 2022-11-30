/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
}

namespace mrc::node {

class EdgeBuilder;

struct EdgeRegistry;

template <typename SourceT, typename SinkT>
struct EdgeBase;

template <typename SourceT, typename SinkT = SourceT, typename EnableT = void>
struct Edge;

template <typename SourceT, typename SinkT>
struct EdgeConnector;

class SinkTypeErased;
class SourceTypeErased;

class SinkPropertiesBase;
class SourcePropertiesBase;
class ObjectPropertiesBase;

template <typename T>
class SinkProperties;

template <typename T>
class SourceProperties;

template <typename ObjectT>
class ObjectProperties;

template <typename T>
class SinkChannel;

template <typename T>
class SourceChannel;

template <typename T>
class RxSinkBase;

template <typename T>
class RxSourceBase;

template <typename T, typename ContextT = runnable::Context>
class RxSink;

template <typename T, typename ContextT = runnable::Context>
class RxSource;

template <typename InputT, typename OutputT = InputT, typename ContextT = runnable::Context>
class RxNode;

class RxSubscribable;

class RxExecute;

template <typename ContextT = runnable::Context>
class RxRunnable;

template <typename T, typename ContextT = runnable::Context>
class GenericSink;

template <typename T, typename ContextT = runnable::Context>
class GenericSource;

template <typename InputT, typename OutputT = InputT, typename ContextT = runnable::Context>
class GenericNode;

}  // namespace mrc::node
