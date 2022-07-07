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

#include "internal/grpc/progress_engine.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "srf/node/forward.hpp"
#include "srf/node/generic_sink.hpp"
#include "srf/node/generic_source.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/runnable/runner.hpp"

#include <boost/fiber/all.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <grpc/support/time.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <grpcpp/support/status_code_enum.h>

#include <chrono>
#include <optional>
#include <tuple>

namespace srf::internal::rpc::server {

using engine_callback_t = std::function<void(bool)>;

template <typename RequestT, typename ResponseT>
class Stream : public Service
{
    using init_fn_t = std::function<void(void* tag)>;

    virtual void on_write_done() {}
    virtual void on_write_fail() {}

    class Manager final : public srf::node::GenericSource<void*>
    {
      public:
        Manager(Stream& parent) : m_parent(parent) {}

      private:
        void data_source(rxcpp::subscriber<void*>& s) final
        {
            boost::fibers::promise<bool> promise_fini;
            // m_context.AsyncNotifyWhenDone(&promise_fini);

            // queue to be initialized
            boost::fibers::promise<bool> promise_init;
            m_parent.m_init_fn(&promise_init);
            auto ok = promise_init.get_future().get();
            if (ok)
            {
                DVLOG(10) << "stream initialized";
            }
            else
            {
                m_parent.m_status_promise.set_value(std::nullopt);
            }
        }

        void did_complete() final
        {
            DVLOG(10) << "stream manager: on_completed - await status";
            auto status = m_parent.m_status_promise.get_future().get();
            if (status)
            {
                Promise<bool> promise;
                DVLOG(10) << "stream manager: on_completed - await status";
                m_parent.m_stream->Finish(*status, &promise);
                promise.get_future().get();
            }
            else
            {
                DVLOG(10) << "no status: stream was likely cancelled before it was initialized; or cancelled before a "
                             "response could be returned";
            }
        }

        Stream& m_parent;
    };

    class Reader final : public srf::node::GenericSource<RequestT>
    {
      public:
        Reader(Stream& parent) : m_parent(parent) {}

      private:
        void data_source(rxcpp::subscriber<RequestT>& s) final
        {
            while (s.is_subscribed())
            {
                boost::fibers::promise<bool> promise;
                m_parent.m_stream->Read(&m_request, &promise);
                auto ok = promise.get_future().get();
                if (ok)
                {
                    DVLOG(10) << "request read: pushing to handler";
                    s.on_next(std::move(m_request));
                }
                else
                {
                    // got WritesDone from client
                    DVLOG(10) << "client issued a WriteDone";
                    s.unsubscribe();
                    m_parent.on_write_done();
                }
            }
        }

        void on_stop(const rxcpp::subscription& subscription) const final{};

        Stream& m_parent;
        RequestT m_request;
    };

    class Writer final : public srf::node::GenericSink<ResponseT>
    {
      public:
        Writer(Stream& parent) : m_parent(parent) {}

      private:
        void on_data(ResponseT&& data) final
        {
            if (m_able_to_write)  // todo(cpp20) [[likely]]
            {
                boost::fibers::promise<bool> promise;
                m_response = std::move(data);
                m_parent.m_stream->Write(m_response, &promise);
                auto ok = promise.get_future().get();
                if (!ok)  // todo(cpp20) [[unlikely]]
                {
                    LOG(WARNING) << "unable to write response to stream";
                    m_able_to_write = false;
                    m_parent.on_write_fail();
                }
            }
        }

        bool m_able_to_write{true};
        ResponseT m_response;
        Stream& m_parent;
        grpc::ServerAsyncReaderWriter<RequestT, ResponseT>& m_stream;
    };

  public:
    using request_fn_t = std::function<void(
        grpc::ServerContext* context, grpc::ServerAsyncReaderWriter<RequestT, ResponseT>* stream, void* tag)>;

    Stream(request_fn_t request_fn, runnable::Resources& runnable) :
      m_runnable(runnable),
      m_stream(std::make_unique<grpc::ServerAsyncReaderWriter<RequestT, ResponseT>>(&m_context))
    {
        m_init_fn = [this, request_fn](void* tag) { request_fn(&m_context, m_stream.get(), tag); };
    }

    void await_write(ResponseT&& response);

  private:
    void do_service_start() final
    {
        m_manager = m_runnable.launch_control().prepare_launcher(std::make_unique<Manager>(*this))->ignition();
        // auto reader  = std::make_unique<Reader>(*this);
        // auto writer  = std::make_unique<Writer>(*this);
    }
    void do_service_stop() final {}
    void do_service_kill() final {}
    void do_service_await_live() final
    {
        m_manager->await_live();
    }
    void do_service_await_join() final
    {
        m_manager->await_join();
    }

    runnable::Resources& m_runnable;

    grpc::ServerContext m_context;

    std::unique_ptr<grpc::ServerAsyncReaderWriter<RequestT, ResponseT>> m_stream;

    init_fn_t m_init_fn;

    // enqueues ResponseT messages to be written to the grpc stream
    std::unique_ptr<srf::node::SourceChannelWriteable<ResponseT>> m_response_channel;

    // pulls RequestT messgaes off the grpc stream and pushes to the handler
    std::unique_ptr<srf::runnable::Runner> m_manager;

    // pulls RequestT messgaes off the grpc stream and pushes to the handler
    std::unique_ptr<srf::runnable::Runner> m_reader;

    // operates on a RequestT message and optionally writes one or more ResponseT message(s)
    std::unique_ptr<srf::runnable::Runner> m_handler;

    // writes ResponseT messages to the grpc stream
    std::unique_ptr<srf::runnable::Runner> m_writer;

    // promise to the optional status object
    Promise<std::optional<grpc::Status>> m_status_promise;

    friend Manager;
    friend Reader;
    friend Writer;
};

template <typename RequestT, typename ResponseT>

class Streaming
{};

}  // namespace srf::internal::rpc::server
