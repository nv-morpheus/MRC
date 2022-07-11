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

#include "srf/channel/channel.hpp"
#include "srf/channel/ingress.hpp"
#include "srf/channel/status.hpp"
#include "srf/node/edge_builder.hpp"
#include "srf/node/edge_properties.hpp"
#include "srf/node/forward.hpp"
#include "srf/node/generic_source.hpp"
#include "srf/node/operators/muxer.hpp"
#include "srf/node/rx_sink.hpp"
#include "srf/node/rx_source.hpp"
#include "srf/node/source_channel.hpp"
#include "srf/runnable/runner.hpp"

#include <boost/fiber/all.hpp>
#include <boost/fiber/future/future_status.hpp>
#include <boost/fiber/operations.hpp>
#include <glog/logging.h>
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

namespace srf::internal::rpc {

using engine_callback_t = std::function<void(bool)>;

template <typename T>
struct StreamWriter : public srf::channel::Ingress<T>
{
    virtual ~StreamWriter()      = default;
    virtual void finish()        = 0;
    virtual void cancel()        = 0;
    virtual bool expired() const = 0;
};

template <typename RequestT, typename ResponseT>
class ClientStreaming : private Service, public std::enable_shared_from_this<ClientStreaming<RequestT, ResponseT>>
{
    using init_fn_t     = std::function<void(void* tag)>;
    using callback_fn_t = std::function<void(const bool&)>;

    using writer_t        = RequestT;
    using reader_t        = ResponseT;
    using stream_writer_t = StreamWriter<writer_t>;

    class ClientStreamWriter final : public stream_writer_t
    {
      public:
        ClientStreamWriter(std::shared_ptr<srf::node::SourceChannelWriteable<writer_t>> channel,
                           std::shared_ptr<ClientStreaming> parent) :
          m_parent(parent),
          m_channel(channel)
        {
            CHECK(parent);
            CHECK(channel);
        }

        srf::channel::Status await_write(writer_t&& t) final
        {
            auto channel = m_channel.lock();
            if (channel)
            {
                return channel->await_write(std::move(t));
            }
            return srf::channel::Status::closed;
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

      private:
        std::weak_ptr<ClientStreaming> m_parent;
        std::weak_ptr<srf::node::SourceChannelWriteable<writer_t>> m_channel;
    };

  public:
    struct IncomingData
    {
        reader_t response;
        std::weak_ptr<stream_writer_t> stream;
    };

    using prepare_fn_t = std::function<std::unique_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>>(
        grpc::ClientContext* context)>;

    ClientStreaming(prepare_fn_t prepare_fn, runnable::Resources& runnable) :
      m_prepare_fn(prepare_fn),
      m_runnable(runnable),
      m_read_channel(std::make_shared<srf::node::Muxer<IncomingData>>())
    {}

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

    void attach_to(srf::node::SinkProperties<IncomingData>& sink)
    {
        srf::node::make_edge(*m_read_channel, sink);
    }

    void attach_to_queue(srf::node::ChannelAcceptor<IncomingData>& sink)
    {
        srf::node::make_edge(*m_read_channel, sink);
    }

  private:
    void do_read(rxcpp::subscriber<IncomingData>& s)
    {
        while (s.is_subscribed())
        {
            CHECK(m_stream);
            Promise<bool> read;
            IncomingData data;
            m_stream->Read(&data.response, &read);
            auto ok = read.get_future().get();
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
                m_can_write = false;
                m_write_channel.reset();
                m_stream_writer.reset();
            }
        }
    }

    void do_writes_done()
    {
        CHECK(m_stream);
        if (m_can_write)
        {
            Promise<bool> writes_done;
            m_stream->WritesDone(&writes_done);
            writes_done.get_future().get();
            DVLOG(10) << "client issued writes done to server";
        };
    }

    void init()
    {
        CHECK(m_stream);
        CHECK(m_read_channel);
        DVLOG(10) << "initializing client stream resources";

        auto reader = std::make_unique<srf::node::RxSource<IncomingData>>(
            rxcpp::observable<>::create<IncomingData>([this](rxcpp::subscriber<IncomingData>& s) {
                do_read(s);
                s.on_completed();
            }));
        srf::node::make_edge(*reader, *m_read_channel);

        // make writer sink
        m_write_channel = std::make_shared<srf::node::SourceChannelWriteable<writer_t>>();
        auto writer     = std::make_unique<srf::node::RxSink<writer_t>>([this](writer_t request) { do_write(request); },
                                                                    [this] { do_writes_done(); });
        srf::node::make_edge(*m_write_channel, *writer);

        // construct StreamWriter
        m_stream_writer = std::shared_ptr<ClientStreamWriter>(
            new ClientStreamWriter(m_write_channel, this->shared_from_this()), [this](ClientStreamWriter* ptr) {
                delete ptr;
                m_write_channel.reset();
                m_stream_writer.reset();
            });

        // launch reader and writer
        m_writer = m_runnable.launch_control().prepare_launcher(std::move(writer))->ignition();
        m_reader = m_runnable.launch_control().prepare_launcher(std::move(reader))->ignition();

        // await live
        m_writer->await_live();
        m_reader->await_live();

        DVLOG(10) << "client stream resources ready";
    }

    void do_service_start() final
    {
        m_stream = m_prepare_fn(&m_context);

        DVLOG(10) << "starting grpc bidi client stream";
        Promise<bool> promise;
        m_stream->StartCall(&promise);
        auto ok = promise.get_future().get();

        if (!ok)
        {
            DVLOG(10) << "grpc bidi client stream - failed to initialize";
            m_stream.reset();
        }

        DVLOG(10) << "grpc bidi client stream - initialized";
        init();
    }

    void do_service_await_live() final {}
    void do_service_stop() final {}
    void do_service_kill() final {}

    void do_service_await_join() final
    {
        m_writer->await_join();
        m_reader->await_join();

        Promise<bool> finish;
        m_stream->Finish(&m_status, &finish);
        auto ok = finish.get_future().get();
    }

    // resources for launching runnables
    runnable::Resources& m_runnable;

    // grpc context
    grpc::ClientContext m_context;

    // grpc reader/writer
    std::shared_ptr<grpc::ClientAsyncReaderWriter<RequestT, ResponseT>> m_stream;

    // prepare fn
    prepare_fn_t m_prepare_fn;

    bool m_can_write{true};
    grpc::Status m_status{grpc::Status::CANCELLED};

    std::shared_ptr<srf::node::SourceChannelWriteable<writer_t>> m_write_channel;
    std::shared_ptr<srf::node::Muxer<IncomingData>> m_read_channel;
    std::shared_ptr<stream_writer_t> m_stream_writer;

    std::unique_ptr<srf::runnable::Runner> m_writer;
    std::unique_ptr<srf::runnable::Runner> m_reader;

    friend ClientStreamWriter;
};

}  // namespace srf::internal::rpc
