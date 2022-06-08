/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include <functional>
#include <memory>

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include "nvrpc/client/base_context.h"
#include "nvrpc/client/executor.h"

namespace nvrpc::client {

namespace detail {

template <template <typename> class Promise, template <typename> class Future,
          typename CompleterFn>
struct shared_packaged_task;

template <template <typename> class Promise, template <typename> class Future,
          typename... Args>
struct shared_packaged_task<Promise, Future, void(Args...)> {
  using CallingFn = std::function<void(Args...)>;
  using WrappedFn = std::function<void(Args...)>;

  shared_packaged_task(CallingFn calling_fn) {
    m_WrappedFn = [this, calling_fn](Args &&...args) {
      calling_fn(args...);
      m_Promise.set_value();
    };
  }

  Future<void> get_future() { return m_Promise.get_future(); }

  void operator()(Args &&...args) { m_WrappedFn(args...); }

private:
  WrappedFn m_WrappedFn;
  Promise<void> m_Promise;
};

template <template <typename> class Promise, template <typename> class Future,
          typename ResultType, typename... Args>
struct shared_packaged_task<Promise, Future, ResultType(Args...)> {
  using CallingFn = std::function<ResultType(Args...)>;
  using WrappedFn = std::function<void(Args...)>;

  shared_packaged_task(CallingFn calling_fn) {
    m_WrappedFn = [this, calling_fn](Args &&...args) {
      m_Promise.set_value(std::move(calling_fn(args...)));
    };
  }

  std::future<ResultType> get_future() { return m_Promise.get_future(); }

  void operator()(Args &&...args) { m_WrappedFn(args...); }

private:
  WrappedFn m_WrappedFn;
  std::promise<ResultType> m_Promise;
};

template <typename CompleterFn> struct async_compute;

template <typename... Args> struct async_compute<void(Args...)> {
  // create a shared object that holds both the promise and the user function
  // to call with some pre-defined arguments.
  // upon calling the () method on the created object, the value of the promise
  // is set by the return value of the wrapped  user function
  template <typename F> static auto wrap(F &&f) {
    using ResultType = typename std::result_of<F(Args...)>::type;
    using UserFn = ResultType(Args...);
    // return std::make_shared<detail::async_compute_impl<UserFn>>(f);
    return std::make_shared<
        shared_packaged_task<std::promise, std::future, UserFn>>(f);
  }
};

} // namespace detail

template <typename Request, typename Response>
struct ClientUnary : public detail::async_compute<void(Request &, Response &,
                                                       ::grpc::Status &)> {
public:
  using PrepareFn = std::function<
      std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>>(
          ::grpc::ClientContext *, const Request &, ::grpc::CompletionQueue *)>;

  ClientUnary(PrepareFn prepare_fn, std::shared_ptr<Executor> executor)
      : m_PrepareFn(prepare_fn), m_Executor(executor) {}

  ~ClientUnary() {}

  template <typename OnReturnFn>
  auto Enqueue(Request *request, Response *response, OnReturnFn on_return,
               std::map<std::string, std::string> &headers) {
    auto wrapped = this->wrap(on_return);
    auto future = wrapped->get_future();

    Context *ctx = new Context;
    ctx->m_Request = request;
    ctx->m_Response = response;
    ctx->m_Callback = [ctx, wrapped]() mutable {
      (*wrapped)(*ctx->m_Request, *ctx->m_Response, ctx->m_Status);
    };

    for (auto &header : headers) {
      ctx->m_Context.AddMetadata(header.first, header.second);
    }

    ctx->m_Reader =
        m_PrepareFn(&ctx->m_Context, *ctx->m_Request, m_Executor->GetNextCQ());
    ctx->m_Reader->StartCall();
    ctx->m_Reader->Finish(ctx->m_Response, &ctx->m_Status, ctx->Tag());

    return future.share();
  }

  template <typename OnReturnFn>
  auto Enqueue(Request &&request, OnReturnFn on_return) {
    std::map<std::string, std::string> empty_headers;
    return Enqueue(std::move(request), on_return, empty_headers);
  }

  template <typename OnReturnFn>
  auto Enqueue(Request &&request, OnReturnFn on_return,
               std::map<std::string, std::string> &headers) {
    auto req = std::make_shared<Request>(std::move(request));
    auto resp = std::make_shared<Response>();

    auto extended_on_return =
        [ req, resp, on_return ](Request & request, Response & response,
                                 ::grpc::Status & status) mutable -> auto {
      return on_return(request, response, status);
    };

    return Enqueue(req.get(), resp.get(), extended_on_return, headers);
  }

private:
  PrepareFn m_PrepareFn;
  std::shared_ptr<Executor> m_Executor;

  class Context : public BaseContext {
    Context() : m_NextState(&Context::StateFinishedDone) {}
    ~Context() override {}

    bool RunNextState(bool ok) final override {
      bool ret = (this->*m_NextState)(ok);
      // DLOG_IF(INFO, !ret) << "RunNextState returning false";
      return ret;
    }

    bool ExecutorShouldDeleteContext() const override { return true; }

  protected:
    bool StateFinishedDone(bool ok) {
      DVLOG(1) << "ClientContext: " << Tag() << " finished with "
               << (m_Status.ok() ? "OK" : "CANCELLED");
      m_Callback();
      DVLOG(1) << "ClientContext: " << Tag() << " callback completed";
      return false;
    }

  private:
    Request *m_Request;
    Response *m_Response;
    std::function<void()> m_Callback;
    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>> m_Reader;
    bool (Context::*m_NextState)(bool);

    friend class ClientUnary;
  };
};

template <typename Request, typename Response>
struct ClientUnaryWithMetaData : public detail::async_compute<void(
                                     Request &, Response &, ::grpc::Status &)> {
public:
  using metadata_t = std::multimap<::grpc::string_ref, ::grpc::string_ref>;

  using PrepareFn = std::function<
      std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>>(
          ::grpc::ClientContext *, const Request &, ::grpc::CompletionQueue *)>;

  ClientUnaryWithMetaData(PrepareFn prepare_fn,
                          std::shared_ptr<Executor> executor)
      : m_PrepareFn(prepare_fn), m_Executor(executor) {}

  ~ClientUnaryWithMetaData() {}

  template <typename OnReturnFn>
  auto Enqueue(Request *request, Response *response, OnReturnFn on_return,
               std::map<std::string, std::string> &headers) {
    auto wrapped = this->wrap(on_return);
    auto future = wrapped->get_future();

    Context *ctx = new Context;
    ctx->m_Request = request;
    ctx->m_Response = response;
    ctx->m_Callback = [ctx, wrapped]() mutable {
      auto metadata = ctx->GetMetaData();
      (*wrapped)(*ctx->m_Request, *ctx->m_Response, ctx->m_Status, metadata);
    };

    for (auto &header : headers) {
      ctx->m_Context.AddMetadata(header.first, header.second);
    }

    ctx->m_Reader =
        m_PrepareFn(&ctx->m_Context, *ctx->m_Request, m_Executor->GetNextCQ());
    ctx->m_Reader->StartCall();
    ctx->m_Reader->Finish(ctx->m_Response, &ctx->m_Status, ctx->Tag());

    return future.share();
  }

  template <typename OnReturnFn>
  auto Enqueue(Request &&request, OnReturnFn on_return) {
    std::map<std::string, std::string> empty_headers;
    return Enqueue(std::move(request), on_return, empty_headers);
  }

  template <typename OnReturnFn>
  auto Enqueue(Request &&request, OnReturnFn on_return,
               std::map<std::string, std::string> &headers) {
    auto req = std::make_shared<Request>(std::move(request));
    auto resp = std::make_shared<Response>();

    auto extended_on_return = [ req, resp, on_return ](
        Request & request, Response & response, ::grpc::Status & status,
        metadata_t & metadata) mutable -> auto {
      return on_return(request, response, status, metadata);
    };

    return Enqueue(req.get(), resp.get(), extended_on_return, headers);
  }

private:
  PrepareFn m_PrepareFn;
  std::shared_ptr<Executor> m_Executor;

  class Context : public BaseContext {
    Context() : m_NextState(&Context::StateFinishedDone) {}
    ~Context() override {}

    bool RunNextState(bool ok) final override {
      bool ret = (this->*m_NextState)(ok);
      // DLOG_IF(INFO, !ret) << "RunNextState returning false";
      return ret;
    }

    bool ExecutorShouldDeleteContext() const override { return true; }

    metadata_t GetMetaData() { return m_Context.GetServerTrailingMetadata(); }

  protected:
    bool StateFinishedDone(bool ok) {
      DVLOG(1) << "ClientContext: " << Tag() << " finished with "
               << (m_Status.ok() ? "OK" : "CANCELLED");
      m_Callback();
      DVLOG(1) << "ClientContext: " << Tag() << " callback completed";
      return false;
    }

  private:
    Request *m_Request;
    Response *m_Response;
    std::function<void()> m_Callback;
    ::grpc::Status m_Status;
    ::grpc::ClientContext m_Context;
    std::unique_ptr<::grpc::ClientAsyncResponseReader<Response>> m_Reader;
    bool (Context::*m_NextState)(bool);

    friend class ClientUnaryWithMetaData;
  };
};

} // namespace nvrpc::client
