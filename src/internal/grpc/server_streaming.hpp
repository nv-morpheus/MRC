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
#include "internal/grpc/stream_writer.hpp"
#include "internal/runnable/resources.hpp"
#include "internal/service.hpp"

#include "mrc/channel/channel.hpp"
#include "mrc/channel/ingress.hpp"
#include "mrc/channel/status.hpp"
#include "mrc/node/edge_builder.hpp"
#include "mrc/node/edge_properties.hpp"
#include "mrc/node/forward.hpp"
#include "mrc/node/generic_source.hpp"
#include "mrc/node/operators/muxer.hpp"
#include "mrc/node/rx_sink.hpp"
#include "mrc/node/rx_source.hpp"
#include "mrc/node/source_channel.hpp"
#include "mrc/runnable/launch_options.hpp"
#include "mrc/runnable/runner.hpp"

#include <boost/fiber/all.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
#include <grpc/grpc_security.h>
#include <grpc/support/time.h>
#include <grpcpp/completion_queue.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>

namespace mrc::internal::rpc {

/**
 * @brief Implementation of a gRPC bidirectional streaming server using MRC primitives
 *
 * ServerStream as three-phases:
 *
 * 1) After construction with a request_fn_t, the ServerStream is "enqueued" with the server by calling `await_init()`.
 *    If `nullptr` is returned, then the stream was not initialized by the server and the object can be destroyed.
 *    Otherwise a shared_ptr to a StreamWriter is returns and the stream is live.
 * 2) The initialization of a live stream creates two MRC runnables:
 *    - A Reader which is a mrc::node::RxSource<IncomingData> and,
 *    - A Writer which is a mrc::node::RxSink<ResponseT>.
 *    These MRC runnables are responsible for pull data off the stream (Reader) and piping IncomingData throught a
 *    user-defined Handler sink which can be attached via the `attach_to` or `attach_to_queue` methods. Part of
 *    IncomingData is a shared_ptr to another instance of StreamWriter allowing the Handler to optional write one or
 *    more responses on to the stream back to the client.
 *    - The StreamWriter is the object which allows responses to be sent. It's lifecycle also maintains the stream.
 *    After the server receives a WritesDone from the client, the ServerStream no longer owns a copy of the
 *    StreamWriter. When the last StreamWriter is destroyed, gRPC issuses a Finish call with a status of OK.
 *    Alternatively, the StreamWriter can early terminate the stream by calling `cancel` or `finish`. Cancel should
 *    return a grpc::Status::CANCELLED status to the client, while Finish would early terminate with the OK status.
 *    - During Phase 2, any holder of StreamWriter can write responses. Typically, the initial StreamWriter from
 *    await_init is copied and owned by one or more server side logic loops which may write response to the client at at
 *    time. StreamWriter can use the expired method to determine if the client-side has issued a WritesDone indicating
 *    that the client has shutdown its sender channel and the server will no longer receieve new client events. This can
 *    be used as a sign for the server to begin shutting down; however, it's not manditory that it does.
 * 3) The final phase is the shutdown phase. There are two indicators of when the Client has issued a WritesDone. As
 *    mentioned above, the StreamWriter::expired is one way to check. The other indication is that the Handler will
 *    receive an IncomingData object with `ok == false`, this will indicate this is the last IncomingData object that
 *    the Handler will process from a given stream. This can be used a direct trigger to any server side logic that the
 *    client has gone quiet. ServerStream::await_fini can be called to ensure the stream is completed on the server
 *    side before being destroyed.
 *
 * @tparam RequestT
 * @tparam ResponseT
 */
template <typename RequestT, typename ResponseT>
class ServerStream : private Service, public std::enable_shared_from_this<ServerStream<RequestT, ResponseT>>
{
    using init_fn_t = std::function<void(void* tag)>;

    using writer_t        = ResponseT;
    using reader_t        = RequestT;
    using stream_writer_t = StreamWriter<writer_t>;

    /**
     * @brief Specialization of StreamWriter for ServerStream
     */
    class ServerStreamWriter final : public stream_writer_t
    {
      public:
        ServerStreamWriter(std::shared_ptr<mrc::node::SourceChannelWriteable<writer_t>> channel,
                           std::shared_ptr<ServerStream> parent) :
          m_parent(parent),
          m_channel(channel)
        {
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

        void finish()
        {
            // todo(ryan) - emplace OK to parent status - possible race condition - might need a mutex
            m_parent->m_write_channel.reset();
        }

        void cancel() final
        {
            // todo(ryan) - emplace only if empty
            m_parent->m_status.emplace(grpc::Status::CANCELLED);
            m_parent->m_context.TryCancel();
        }

        bool expired() const final
        {
            return false;  // todo(ryan) - could be m_channel.expired();
        }

        std::size_t get_id() const final
        {
            return reinterpret_cast<std::size_t>(m_parent.get());
        }

      private:
        const std::shared_ptr<ServerStream<RequestT, ResponseT>> m_parent;
        std::weak_ptr<mrc::node::SourceChannelWriteable<writer_t>> m_channel;
    };

  public:
    struct IncomingData
    {
        reader_t msg;
        std::shared_ptr<stream_writer_t> stream;
        bool ok;
    };

    using request_fn_t = std::function<void(
        grpc::ServerContext* context, grpc::ServerAsyncReaderWriter<ResponseT, RequestT>* stream, void* tag)>;

    ServerStream(request_fn_t request_fn, runnable::Resources& runnable) :
      m_runnable(runnable),
      m_stream(std::make_unique<grpc::ServerAsyncReaderWriter<ResponseT, RequestT>>(&m_context)),
      m_reader_source(std::make_unique<mrc::node::RxSource<IncomingData>>(
          rxcpp::observable<>::create<IncomingData>([this](rxcpp::subscriber<IncomingData> s) {
              this->do_read(s);
              s.on_completed();
          })))
    {
        m_init_fn = [this, request_fn](void* tag) { request_fn(&m_context, m_stream.get(), tag); };
    }

    ~ServerStream() override
    {
        Service::call_in_destructor();
    }

    std::size_t get_id() const
    {
        return reinterpret_cast<std::size_t>(this);
    }

    inline std::shared_ptr<stream_writer_t> writer() const
    {
        return m_weak_stream_writer.lock();
    }

    std::shared_ptr<stream_writer_t> await_init()
    {
        // make this only callable once
        service_start();
        return writer();
    }

    grpc::Status await_fini()
    {
        service_await_join();
        if (m_status)
        {
            return *m_status;
        }
        return grpc::Status::CANCELLED;
    }

    // must be called before await_init()
    void attach_to(mrc::node::SinkProperties<IncomingData>& sink)
    {
        CHECK(m_reader_source);
        mrc::node::make_edge(*m_reader_source, sink);
    }

    // must be called before await_init()
    void attach_to_queue(mrc::node::ChannelAcceptor<IncomingData>& queue)
    {
        CHECK(m_reader_source);
        mrc::node::make_edge(*m_reader_source, queue);
    }

  private:
    // logic executed by the Reader
    void do_read(rxcpp::subscriber<IncomingData>& s)
    {
        while (s.is_subscribed())
        {
            CHECK(m_stream);
            Promise<bool> read;
            IncomingData data;
            m_stream->Read(&data.msg, &read);
            auto ok     = read.get_future().get();
            data.ok     = ok;
            data.stream = writer();
            s.on_next(std::move(data));
            if (!ok)
            {
                // client issued a writes done
                // this information was forwarded to the handler via data.ok
                DVLOG(10) << "server got writes done from client - server reads now done";
                m_stream_writer.reset();
                return;
            }
        }
    }

    // logic executed by the Writer's on_next method
    void do_write(const writer_t& request)
    {
        CHECK(m_stream);
        if (m_can_write)
        {
            Promise<bool> promise;
            m_stream->Write(request, &promise);
            auto ok = promise.get_future().get();
            if (!ok)
            {
                DVLOG(10) << "server failed to write to client; disabling writes and beginning shutdown";
                m_can_write = false;
                m_write_channel.reset();
                m_stream_writer.reset();
                m_context.TryCancel();
            }
        }
    }

    // logic executed in the Writers on_completed
    void do_writes_done()
    {
        if (m_can_write)
        {
            if (!m_status)
            {
                m_status.emplace(grpc::Status::OK);
            }

            DVLOG(10) << "server issuing finish";
            Promise<bool> finish;
            m_stream->Finish(*m_status, &finish);
            auto ok = finish.get_future().get();
            DVLOG(10) << "server done with finish";
        }
    }

    // this method is executed if the grpc stream is successfully initialized
    void do_init()
    {
        CHECK(m_stream);
        CHECK(m_reader_source);

        // make writer sink
        m_write_channel = std::make_shared<mrc::node::SourceChannelWriteable<writer_t>>();
        auto writer     = std::make_unique<mrc::node::RxSink<writer_t>>([this](writer_t request) { do_write(request); },
                                                                    [this] { do_writes_done(); });
        mrc::node::make_edge(*m_write_channel, *writer);

        // construct StreamWriter
        m_can_write     = true;
        m_stream_writer = std::shared_ptr<ServerStreamWriter>(
            new ServerStreamWriter(m_write_channel, this->shared_from_this()), [this](ServerStreamWriter* ptr) {
                delete ptr;
                m_write_channel.reset();
            });
        m_weak_stream_writer = m_stream_writer;

        // launch reader and writer
        m_writer = m_runnable.launch_control().prepare_launcher(std::move(writer))->ignition();
        m_reader = m_runnable.launch_control().prepare_launcher(std::move(m_reader_source))->ignition();

        // await live
        m_writer->await_live();
        m_reader->await_live();
    }

    void do_service_start() final
    {
        Promise<bool> promise;
        m_init_fn(&promise);
        auto ok = promise.get_future().get();

        if (!ok)
        {
            DVLOG(10) << "server stream could not be initialized";
            m_stream.reset();
            m_reader_source.reset();
            // this marks the Service as joined - no need to join it in this state
            service_await_join();
            return;
        }

        do_init();
    }

    void do_service_await_live() final {}
    void do_service_stop() final
    {
        m_context.TryCancel();
    }
    void do_service_kill() final
    {
        m_context.TryCancel();
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
    runnable::Resources& m_runnable;

    // grpc context
    grpc::ServerContext m_context;

    // grpc reader/writer
    std::shared_ptr<grpc::ServerAsyncReaderWriter<ResponseT, RequestT>> m_stream;

    // prepare fn
    init_fn_t m_init_fn;

    // state variables
    bool m_can_write{false};  // false indicates the write stream is not operational
    std::optional<grpc::Status> m_status;

    // reader_source - available for handler connections (attach_to method) prior to await_init
    std::unique_ptr<mrc::node::RxSource<IncomingData>> m_reader_source;

    // channel connected to the writer sink; each ServerStreamWriter will take ownership of a shared_ptr
    std::shared_ptr<mrc::node::SourceChannelWriteable<writer_t>> m_write_channel;

    // the destruction of this object also ensures that m_write_channel is reset
    // this object is nullified after the last IncomingData object is passed to the handler
    std::shared_ptr<stream_writer_t> m_stream_writer;
    std::weak_ptr<stream_writer_t> m_weak_stream_writer;

    // runners to manage the life cycles of the reader / writer runnables
    std::unique_ptr<mrc::runnable::Runner> m_writer;
    std::unique_ptr<mrc::runnable::Runner> m_reader;

    friend ServerStreamWriter;
};

}  // namespace mrc::internal::rpc
