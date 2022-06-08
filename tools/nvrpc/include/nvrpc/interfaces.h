/**
 * SPDX-FileCopyrightText: Copyright (c) 2018-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NVIS_INTERFACES_H_
#define NVIS_INTERFACES_H_

#include <grpcpp/grpcpp.h>

namespace nvrpc {

class IContext;
class IExecutor;
class IContextLifeCycle;
class IRPC;
class IService;

struct Resources : public std::enable_shared_from_this<Resources>
{
    virtual ~Resources() = default;

    template <class Target>  // NOLINT
    std::shared_ptr<Target> casted_shared_from_this()
    {
        return std::dynamic_pointer_cast<Target>(Resources::shared_from_this());
    }
};

/**
 * The IContext object and it's subsequent derivations are the single more important class
 * in this library. Contexts are responsible for maintaining the state of a message and
 * performing the custom code for an RPC invocation.
 */
class IContext
{
  public:
    virtual ~IContext() {}
    static IContext* Detag(void* tag)
    {
        return static_cast<IContext*>(tag);
    }

  protected:
    IContext() : m_PrimaryContext(this) {}
    IContext(IContext* primary_context) : m_PrimaryContext(primary_context) {}

    void* Tag()
    {
        return reinterpret_cast<void*>(this);
    }

  protected:
    IContext* m_PrimaryContext;

  private:
    virtual bool RunNextState(bool) = 0;
    virtual void Reset()            = 0;

    friend class IRPC;
    friend class IExecutor;
};

class IContextLifeCycle : public IContext
{
  public:
    ~IContextLifeCycle() override {}

  protected:
    IContextLifeCycle() = default;

    virtual void OnLifeCycleStart() = 0;
    virtual void OnLifeCycleReset() = 0;

    virtual void FinishResponse() = 0;
    virtual void CancelResponse() = 0;
};

class IService
{
  public:
    IService() = default;
    virtual ~IService() {}

    virtual void Initialize(::grpc::ServerBuilder&) = 0;
};

class IRPC
{
  public:
    IRPC() = default;
    virtual ~IRPC() {}

  protected:
    virtual std::unique_ptr<IContext> CreateContext(::grpc::ServerCompletionQueue*, std::shared_ptr<Resources>) = 0;

    friend class IExecutor;
};

class IExecutor
{
  public:
    IExecutor() = default;
    virtual ~IExecutor() {}

    virtual void Initialize(::grpc::ServerBuilder&)                                                          = 0;
    virtual void Run()                                                                                       = 0;
    virtual void RegisterContexts(IRPC* rpc, std::shared_ptr<Resources> resources, int numContextsPerThread) = 0;
    virtual void Shutdown()                                                                                  = 0;

  protected:
    using time_point = std::chrono::system_clock::time_point;

    virtual void SetTimeout(time_point, std::function<void()>) = 0;

    inline bool RunContext(IContext* ctx, bool ok)
    {
        return ctx->RunNextState(ok);
    }
    inline void ResetContext(IContext* ctx)
    {
        ctx->Reset();
    }
    inline std::unique_ptr<IContext> CreateContext(IRPC* rpc,
                                                   ::grpc::ServerCompletionQueue* cq,
                                                   std::shared_ptr<Resources> res)
    {
        return rpc->CreateContext(cq, res);
    }
};

}  // namespace nvrpc

#endif  // NVIS_INTERFACES_H_
