/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace mrc::edge {

template <typename T>
class Edge;

class EdgeBuilder;

template <typename SourceT, typename SinkT>
struct EdgeConnector;

template <typename T>
class EdgeChannelReader;

template <typename T>
class EdgeChannelWriter;

template <typename T>
class EdgeChannel;

template <typename T>
class EdgeHolder;

template <typename KeyT, typename T>
class MultiEdgeHolder;

class EdgeTypeInfo;

class IEdgeReadableBase;

template <typename T>
class IEdgeReadable;

class ReadableEdgeHandle;

class IReadableProviderBase;

class IReadableAcceptorBase;

template <typename T>
class IReadableProvider;

template <typename T>
class IReadableAcceptor;

class IEdgeWritableBase;

template <typename T>
class IEdgeWritable;

class WritableEdgeHandle;

class DeferredWritableEdgeHandle;

class IWritableProviderBase;

class IWritableAcceptorBase;

template <typename KeyT>
class IMultiWritableAcceptorBase;

template <typename T>
class IWritableProvider;

template <typename T>
class IWritableAcceptor;

template <typename T, typename KeyT>
class IMultiWritableAcceptor;

}  // namespace mrc::edge
