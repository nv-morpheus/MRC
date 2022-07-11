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
#include "srf/runnable/launcher.hpp"
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
#include <memory>
#include <optional>
#include <tuple>

namespace srf::internal::rpc {

template <typename ReadT, typename WriteT>
class AsyncStreamingContext : public Service
{
  public:
  private:
    class Handler final : public srf::node::GenericNode<ReadT, WriteT>
    {
      public:
        Handler(AsyncStreamingContext& parent) : m_parent(parent), m_stream(m_parent.m_write_channel) {}

      private:
        void on_data(ReadT&& data, rxcpp::subscriber<WriteT>& subscriber) final
        {
            m_parent.on_read(std::move(data), m_stream);
        }

        AsyncStreamingContext& m_parent;
        AsyncStream<WriteT> m_stream;
    };

  public:
    AsyncStreamingContext(runnable::Resources& runnable) : m_runnable(runnable) {}

    ~AsyncStreamingContext() override
    {
        Service::call_in_destructor();
    }

    bool await_fini()
    {
        for (auto& runner : m_runners)
        {
            runner->await_join();
        }
        bool ok;
        auto rc = m_status_channel->egress().await_read(ok);
        if (ok && rc == srf::channel::Status::success)
        {
            ok = do_fini(grpc::Status::OK);
        }
        return ok;
    }

  protected:
    void* init_tag()
    {
        return &m_init_promise;
    }

    virtual void on_reader_complete(bool ok) {}
    virtual void on_init(std::optional<AsyncStream<WriteT>> stream) {}
    virtual void on_fini(bool ok) {}

    void do_service_start() override
    {
        bool status = m_init_promise.get_future().get();
        std::optional<AsyncStream<WriteT>> stream;
        if (status)
        {
            m_status_channel = std::make_unique<srf::node::SinkChannelReadable<bool>>();
            m_write_channel  = std::make_shared<srf::node::SourceChannelWriteable<WriteT>>();

            auto reader  = std::make_unique<Reader<ReadT>>(async_reader());
            auto writer  = std::make_unique<Writer<WriteT>>(async_writer());
            auto handler = std::make_unique<Handler>(*this);

            srf::node::make_edge(*reader, *handler);
            srf::node::make_edge(*handler, *writer);
            srf::node::make_edge(*m_write_channel, *writer);
            srf::node::make_edge(*writer, *m_status_channel);

            std::vector<std::unique_ptr<srf::runnable::Launcher>> launchers;
            launchers.push_back(m_runnable.launch_control().prepare_launcher(std::move(writer)));
            launchers.push_back(m_runnable.launch_control().prepare_launcher(std::move(handler)));
            launchers.push_back(m_runnable.launch_control().prepare_launcher(std::move(reader)));

            launchers.at(2)->apply([this](srf::runnable::Runner& runner) {
                runner.on_completion_callback([this](bool ok) { on_reader_complete(ok); });
            });

            for (auto& launcher : launchers)
            {
                m_runners.push_back(std::move(launcher->ignition()));
            }

            stream.emplace(m_write_channel);
        }
        on_init(stream);
    }

  private:
    virtual bool do_fini(grpc::Status status)                                            = 0;
    virtual void on_read(ReadT&& request, const AsyncStream<WriteT>& stream)             = 0;
    virtual std::shared_ptr<grpc::internal::AsyncReaderInterface<ReadT>> async_reader()  = 0;
    virtual std::shared_ptr<grpc::internal::AsyncWriterInterface<WriteT>> async_writer() = 0;

    void do_service_stop() final {}
    void do_service_kill() final {}
    void do_service_await_live() final
    {
        for (auto& runner : m_runners)
        {
            runner->await_live();
        }
    }
    void do_service_await_join() final
    {
        for (auto& runner : m_runners)
        {
            runner->await_join();
        }
        bool ok = false;
        if (m_status_channel)
        {
            auto rc = m_status_channel->egress().await_read(ok);
            if (ok && rc == srf::channel::Status::success)
            {
                ok = do_fini(grpc::Status::OK);
            }
        }
        on_fini(ok);
    }

    runnable::Resources& m_runnable;
    Promise<bool> m_init_promise;
    std::unique_ptr<srf::node::SinkChannelReadable<bool>> m_status_channel;
    std::shared_ptr<srf::node::SourceChannelWriteable<WriteT>> m_write_channel;
    std::vector<std::unique_ptr<srf::runnable::Runner>> m_runners;
};

}  // namespace srf::internal::rpc
