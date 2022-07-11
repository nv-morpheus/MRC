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

#include "internal/grpc/async_reader.hpp"
#include "internal/grpc/async_stream.hpp"
#include "internal/grpc/async_streaming_context.hpp"
#include "internal/grpc/async_writer.hpp"
#include "internal/grpc/progress_engine.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "srf/channel/status.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/generic_node.hpp"
#include "srf/node/generic_sink.hpp"
#include "srf/node/generic_source.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/runnable/runner.hpp"

#include <boost/fiber/all.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <grpc/support/time.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <chrono>
#include <memory>
#include <optional>
#include <tuple>

namespace srf::internal::rpc::client {

using engine_callback_t = std::function<void(bool)>;

template <typename RequestT, typename ResponseT>
class ClientAsyncStreamingContext : public AsyncStreamingContext<ResponseT, RequestT>
{
    using init_fn_t = std::function<void(void* tag)>;
    using base_t  = AsyncStreamingContext<ResponseT, RequestT>;

  public:
    using prepare_fn_t = std::function<std::unique_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>>(
        grpc::ClientContext* context)>;

    ClientAsyncStreamingContext(prepare_fn_t prepare_fn, runnable::Resources& runnable) : base_t(runnable), m_prepare_fn(prepare_fn)
    {}

    ~ClientAsyncStreamingContext() override {}

  private:
    void do_service_start() final
    {
        m_stream = m_prepare_fn(&m_context);
        m_stream->StartCall(this->init_tag());
        // AsyncStreamingContext will await the init tag completion
        base_t::do_service_start();
    }

    bool do_fini(grpc::Status status) final
    {
        Promise<bool> promise;
        m_stream->Finish(&status, &promise);
        return promise.get_future().get();
    }

    std::shared_ptr<grpc::internal::AsyncReaderInterface<ResponseT>> async_reader() final
    {
        return m_stream;
    }
    std::shared_ptr<grpc::internal::AsyncWriterInterface<RequestT>> async_writer() final
    {
        return m_stream;
    }

    // grpc context
    grpc::ClientContext m_context;

    // grpc reader/writer
    std::shared_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>> m_stream;

    // prepare fn
    prepare_fn_t m_prepare_fn;
};

}  // namespace srf::internal::rpc::client
