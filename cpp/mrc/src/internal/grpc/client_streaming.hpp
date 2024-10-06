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

#pragma once

#include "internal/grpc/progress_engine.hpp"
#include "internal/grpc/promise_handler.hpp"
#include "internal/grpc/stream_writer.hpp"
#include "internal/runnable/runnable_resources.hpp"
#include "internal/service.hpp"

#include "mrc/channel/channel.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/edge/edge_builder.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/source_channel_owner.hpp"
#include "mrc/node/writable_entrypoint.hpp"
#include "mrc/runnable/runner.hpp"

#include <boost/fiber/all.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <grpcpp/client_context.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>

namespace mrc::rpc {

/**
 * @brief Implementation of a gRPC bidirectional streaming client using MRC primitives
 *
 * The client mimics the server with both reader and writer runnables, but its StreamWriter (ClientStreamWriter)
 * lifespan controls issues a WritesDone on destruction.
 *
 * Similar to ServerStream, ClientStream operates in three phases:
 * 1) On construction and upto calling await_init, a Node/Sink<IncomingData> can be attached to the reader. On
 * await_init, if the stream fails to initialize a nullptr is returned and the ClientStream object can be destroyed.
 * 2) Otherwise, await_init returns a share_ptr to a StreamWriter who lifecycle is tied to the gRPC async writer. When
 * the final StreamWriter is release, a WritesDone is issues to the server and no more writes can be issued. At this
 * point the Writer runnable will be completed.
 * Similar to ServerStream, incoming ResponseT messages from the server will be routed to the connected Handler
 * with a IncomingData object that contains both the response message and an instance of the StreamWriter.
 * 3) Finally, after a WritesDone is issued, the Reader will stay alive until the server closes it; at which the Finish
 * method will be observed on await_fini.
 *
 * Early termination on the server will shutdown the Readers and Writers and will disconnect the StreamWriter from being
 * able to access the parent or the write channel.
 */
template <typename RequestT, typename ResponseT>
class ClientStream : private Service, public std::enable_shared_from_this<ClientStream<RequestT, ResponseT>>
{
    using init_fn_t     = std::function<void(void* tag)>;
    using callback_fn_t = std::function<void(const bool&)>;

    using writer_t        = RequestT;
    using reader_t        = ResponseT;
    using stream_writer_t = StreamWriter<writer_t>;

    class ClientStreamWriter final : public stream_writer_t
    {
      public:
        ClientStreamWriter(std::shared_ptr<mrc::node::WritableEntrypoint<writer_t>> channel,
                           std::shared_ptr<ClientStream> parent) :
          m_parent(parent),
          m_channel(channel)
        {
            CHECK(parent);
            CHECK(channel);
        }

        mrc::channel::Status await_write(writer_t&& t) final
        {
            auto channel = m_channel.lock();
            if (channel)
            {
                return channel->await_write(std::move(t));
            }
            return mrc::channel::Status::closed;
        }

        void finish() final
        {
            auto parent = m_parent.lock();
            if (parent)
            {
                parent->m_write_channel.reset();
            }
        }

        void cancel() final
        {
            auto parent = m_parent.lock();
            if (parent)
            {
                parent->m_context.TryCancel();
            }
        }

        bool expired() const final
        {
            return m_parent.expired();
        }

        std::size_t get_id() const final
        {
            auto parent = m_parent.lock();
            if (parent)
            {
                return reinterpret_cast<std::size_t>(parent.get());
            }
            return 0UL;
        }

      private:
        std::weak_ptr<ClientStream> m_parent;
        std::weak_ptr<mrc::node::WritableEntrypoint<writer_t>> m_channel;
    };

  public:
    struct IncomingData
    {
        reader_t msg;
        std::weak_ptr<stream_writer_t> stream;
    };

    using prepare_fn_t =
        std::function<std::unique_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>>(grpc::ClientContext* context)>;

    ClientStream(prepare_fn_t prepare_fn, runnable::RunnableResources& runnable) :
      Service("rpc::ClientStream"),
      m_prepare_fn(prepare_fn),
      m_runnable(runnable),
      m_reader_source(std::make_unique<mrc::node::RxSource<IncomingData>>(
          rxcpp::observable<>::create<IncomingData>([this](rxcpp::subscriber<IncomingData>& s) {
              do_read(s);
              s.on_completed();
          })))
    {}

    ~ClientStream() override
    {
        Service::call_in_destructor();
    }

    std::shared_ptr<stream_writer_t> await_init()
    {
        // make this only callable once
        service_start();
        return m_stream_writer;
    }

    grpc::Status await_fini()
    {
        service_await_join();
        return m_status;
    }

    // todo(ryan) - add a method to trigger a writes done

    template <typename NodeT>
    void attach_to(NodeT& sink)
    {
        CHECK(m_reader_source);
        mrc::make_edge(*m_reader_source, sink);
    }

  private:
    // logic performed by the Reader Runnable
    void do_read(rxcpp::subscriber<IncomingData>& s)
    {
        while (s.is_subscribed())
        {
            CHECK(m_stream);
            auto* wrapper = new PromiseWrapper("Client::Read");
            IncomingData data;
            m_stream->Read(&data.msg, wrapper);
            auto ok = wrapper->get_future();
            if (!ok)
            {
                m_write_channel.reset();
                m_stream_writer.reset();
                return;
            }
            data.stream = m_stream_writer;
            s.on_next(std::move(data));
        }
    }

    // logic performed by the Writer Runnable
    void do_write(const writer_t& request)
    {
        CHECK(m_stream);
        if (m_can_write)
        {
            auto* wrapper = new PromiseWrapper("Client::Write");
            m_stream->Write(request, wrapper);
            auto ok = wrapper->get_future();
            if (!ok)
            {
                m_can_write = false;
                m_write_channel.reset();
                m_stream_writer.reset();
            }
        }
    }

    // logic performed on the Writer's on_completed
    void do_writes_done()
    {
        CHECK(m_stream);
        if (m_can_write)
        {
            {
                auto* wrapper = new PromiseWrapper("Client::WritesDone");
                m_stream->WritesDone(wrapper);
                wrapper->get_future();
            }

            {
                // Now issue finish since this is OK at the client level
                auto* wrapper = new PromiseWrapper("Client::Finish");
                m_stream->Finish(&m_status, wrapper);
                wrapper->get_future();
            }

            // DVLOG(10) << "client issued writes done to server";
        };
    }

    // initialization performed after the grpc client stream was successfully initialized
    void do_init()
    {
        CHECK(m_stream);
        CHECK(m_reader_source);
        DVLOG(10) << "initializing client stream resources";

        // make writer sink
        m_write_channel = std::make_shared<mrc::node::WritableEntrypoint<writer_t>>();
        auto writer     = std::make_unique<mrc::node::RxSink<writer_t>>(
            [this](writer_t request) {
                do_write(request);
            },
            [this] {
                do_writes_done();
            });
        mrc::make_edge(*m_write_channel, *writer);

        // construct StreamWriter
        m_stream_writer = std::shared_ptr<ClientStreamWriter>(
            new ClientStreamWriter(m_write_channel, this->shared_from_this()),
            [this](ClientStreamWriter* ptr) {
                delete ptr;
                m_write_channel.reset();
                m_stream_writer.reset();
            });

        // launch reader and writer
        m_writer = m_runnable.launch_control().prepare_launcher(std::move(writer))->ignition();
        m_reader = m_runnable.launch_control().prepare_launcher(std::move(m_reader_source))->ignition();

        // await live
        m_writer->await_live();
        m_reader->await_live();

        DVLOG(10) << "client stream resources ready";
    }

    void do_service_start() final
    {
        m_stream = m_prepare_fn(&m_context);

        DVLOG(10) << "starting grpc bidi client stream";
        auto* wrapper = new PromiseWrapper("Client::StartCall", false);
        m_stream->StartCall(wrapper);
        auto ok = wrapper->get_future();

        if (!ok)
        {
            DVLOG(10) << "grpc bidi client stream - failed to initialize";
            m_stream.reset();
            m_reader_source.reset();
            service_await_join();
            return;
        }

        DVLOG(10) << "grpc bidi client stream - initialized";
        do_init();
    }

    void do_service_await_live() final
    {
        m_reader->await_live();
        m_writer->await_live();
    }

    void do_service_stop() final
    {
        m_stream_writer->finish();
        m_stream_writer.reset();
    }

    void do_service_kill() final
    {
        m_stream_writer->finish();
        m_stream_writer.reset();
    }

    void do_service_await_join() final
    {
        if (m_writer || m_reader)
        {
            CHECK(m_writer && m_reader);

            m_writer->await_join();
            m_reader->await_join();
        }
    }

    // resources for launching runnables
    runnable::RunnableResources& m_runnable;

    // grpc context
    grpc::ClientContext m_context;

    // grpc reader/writer
    std::shared_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>> m_stream;

    // prepare fn
    prepare_fn_t m_prepare_fn;

    // state variables
    bool m_can_write{true};
    grpc::Status m_status{grpc::Status::CANCELLED};

    // holder of the reader source until the grpc stream goes live
    std::unique_ptr<mrc::node::RxSource<IncomingData>> m_reader_source;

    // channel connected to the writer sink; each ServerStreamWriter will take ownership of a shared_ptr
    std::shared_ptr<mrc::node::WritableEntrypoint<writer_t>> m_write_channel;

    // the destruction of this object also ensures that m_write_channel is reset
    // this object is nullified after the last IncomingData object is passed to the handler
    std::shared_ptr<stream_writer_t> m_stream_writer;

    // runners to manage the life cycles of the reader / writer runnables
    std::unique_ptr<mrc::runnable::Runner> m_writer;
    std::unique_ptr<mrc::runnable::Runner> m_reader;

    friend ClientStreamWriter;
};

}  // namespace mrc::rpc
