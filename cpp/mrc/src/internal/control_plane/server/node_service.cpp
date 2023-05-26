/**
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

#include "internal/control_plane/server/node_service.hpp"

#include "mrc/core/async_service.hpp"
#include "mrc/protos/architect.grpc.pb.h"
#include "mrc/protos/architect.pb.h"
#include "mrc/utils/library_utils.hpp"
#include "mrc/utils/string_utils.hpp"

#include <grpcpp/client_context.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/support/status.h>
#include <node/node.h>
#include <uv.h>
#include <v8-initialization.h>

#include <algorithm>
#include <exception>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

namespace mrc::control_plane {

std::vector<char*> vec_string_to_char_ptr(std::vector<std::string>& vec_strings)
{
    std::vector<char*> vec_pointers;

    // remember the nullptr terminator
    vec_pointers.reserve(vec_strings.size() + 1);

    std::transform(vec_strings.begin(), vec_strings.end(), std::back_inserter(vec_pointers), [](std::string& s) {
        return s.data();
    });

    vec_pointers.push_back(nullptr);

    return vec_pointers;
}

// int run_node_instance(::node::MultiIsolatePlatform* platform,
//                       const std::vector<std::string>& args,
//                       const std::vector<std::string>& exec_args)
// {
//     int exit_code = 0;

//     std::vector<std::string> errors;
//     std::unique_ptr<::node::CommonEnvironmentSetup> setup =
//         ::node::CommonEnvironmentSetup::Create(platform, &errors, args, exec_args);

//     if (!setup)
//     {
//         for (const std::string& err : errors)
//         {
//             fprintf(stderr, "%s: %s\n", args[0].c_str(), err.c_str());
//         }
//         return 1;
//     }

//     v8::Isolate* isolate     = setup->isolate();
//     ::node::Environment* env = setup->env();

//     {
//         v8::Locker locker(isolate);
//         v8::Isolate::Scope isolate_scope(isolate);
//         v8::HandleScope handle_scope(isolate);
//         v8::Context::Scope context_scope(setup->context());

//         // auto loadenv_ret = ::node::LoadEnvironment(env,
//         //                                            "process.stdout.write('test');"
//         //                                            "const publicRequire ="
//         //                                            "  require('module').createRequire(process.cwd() + '/');"
//         //                                            "globalThis.require = publicRequire;"
//         //                                            "require('vm').runInThisContext(process.argv[1]);");

//         auto loadenv_ret = ::node::LoadEnvironment(env, ::node::StartExecutionCallback{});

//         if (loadenv_ret.IsEmpty())
//         {  // There has been a JS exception.
//             return 1;
//         }

//         exit_code = ::node::SpinEventLoop(env).FromMaybe(1);

//         ::node::Stop(env);
//     }

//     return exit_code;
// }

// int run_node(std::vector<std::string> args)
// {
//     // Convert the string array to a char**
//     std::vector<char*> raw_args_array = vec_string_to_char_ptr(args);

//     char** argv = uv_setup_args(args.size(), raw_args_array.data());

//     // Parse Node.js CLI options, and print any errors that have occurred while
//     // trying to parse them.
//     std::unique_ptr<::node::InitializationResult> result = ::node::InitializeOncePerProcess(
//         args,
//         {::node::ProcessInitializationFlags::kNoInitializeV8,
//          ::node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

//     for (const std::string& error : result->errors())
//     {
//         fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());
//     }

//     if (static_cast<int>(result->early_return()) != 0)
//     {
//         // return result->exit_code();
//         LOG(ERROR) << "Exited early. Code: " << result->exit_code();
//     }

//     // Create a v8::Platform instance. `MultiIsolatePlatform::Create()` is a way
//     // to create a v8::Platform instance that Node.js can use when creating
//     // Worker threads. When no `MultiIsolatePlatform` instance is present,
//     // Worker threads are disabled.
//     std::unique_ptr<::node::MultiIsolatePlatform> platform = ::node::MultiIsolatePlatform::Create(4);
//     v8::V8::InitializePlatform(platform.get());
//     v8::V8::Initialize();

//     // See below for the contents of this function.
//     int ret = run_node_instance(platform.get(), result->args(), result->exec_args());

//     v8::V8::Dispose();
//     v8::V8::DisposePlatform();

//     ::node::TearDownOncePerProcess();

//     return ret;
// }

// // This should happen once per process
// std::unique_ptr<::node::MultiIsolatePlatform> node_init_platform(std::vector<std::string>& args)
// {
//     // Convert the string array to a char**
//     std::vector<char*> raw_args_array = vec_string_to_char_ptr(args);

//     // Initialize uv with the arguments
//     char** argv = uv_setup_args(args.size(), raw_args_array.data());

//     // Update the args (I gues UV can change them?)
//     args = std::vector<std::string>(argv, argv + args.size());

//     // Parse Node.js CLI options, and print any errors that have occurred while
//     // trying to parse them.
//     std::unique_ptr<::node::InitializationResult> result = ::node::InitializeOncePerProcess(
//         args,
//         {::node::ProcessInitializationFlags::kNoInitializeV8,
//          ::node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

//     for (const std::string& error : result->errors())
//     {
//         fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());
//     }

//     if (static_cast<int>(result->early_return()) != 0)
//     {
//         // return result->exit_code();
//         LOG(ERROR) << "Exited early. Code: " << result->exit_code();
//         return nullptr;
//     }

//     // Create a v8::Platform instance. `MultiIsolatePlatform::Create()` is a way
//     // to create a v8::Platform instance that Node.js can use when creating
//     // Worker threads. When no `MultiIsolatePlatform` instance is present,
//     // Worker threads are disabled.
//     std::unique_ptr<::node::MultiIsolatePlatform> platform = ::node::MultiIsolatePlatform::Create(4);
//     v8::V8::InitializePlatform(platform.get());
//     v8::V8::Initialize();

//     return platform;
// }

// // std::unique_ptr<::node::CommonEnvironmentSetup> node_init_setup(std::vector<std::string>& args) {

// // }

// int node_run_environment(const std::unique_ptr<::node::CommonEnvironmentSetup>& setup)
// {
//     int exit_code = 0;

//     v8::Isolate* isolate     = setup->isolate();
//     ::node::Environment* env = setup->env();

//     {
//         v8::Locker locker(isolate);
//         v8::Isolate::Scope isolate_scope(isolate);
//         v8::HandleScope handle_scope(isolate);
//         v8::Context::Scope context_scope(setup->context());

//         auto loadenv_ret = ::node::LoadEnvironment(env, ::node::StartExecutionCallback{});

//         if (loadenv_ret.IsEmpty())
//         {  // There has been a JS exception.
//             return 1;
//         }

//         exit_code = ::node::SpinEventLoop(env).FromMaybe(1);

//         ::node::Stop(env);
//     }

//     return exit_code;
// }

// void NodeContext::do_init() {}

// NodeRuntime::NodeRuntime()  = default;
// NodeRuntime::~NodeRuntime() = default;

// void NodeRuntime::run(::mrc::runnable::Context& ctx)
// {
//     std::vector<std::string> args;

//     args.emplace_back("/work/build/cpp/mrc/src/tests/test_mrc_private.x");
//     args.emplace_back("--inspect");
//     args.emplace_back("/work/ts/control-plane/dist/server/server.js");
//     // args.emplace_back("/work/cpp/mrc/src/internal/control_plane/server.js");

//     // Now start node
//     run_node(args);

//     // Create the setup object to allow us to stop in the future
//     this->launch_node(args);
// }

// void NodeRuntime::on_state_update(const Runnable::State& state)
// {
//     switch (state)
//     {
//     case Runnable::State::Stop: {
//         // Send a gRPC message to shutdown the server
//         auto channel = grpc::CreateChannel("localhost:4000", grpc::InsecureChannelCredentials());
//         auto stub    = mrc::protos::Architect::NewStub(channel);

//         auto context = grpc::ClientContext();

//         ::mrc::protos::ShutdownRequest request;
//         ::mrc::protos::ShutdownResponse response;

//         stub->Shutdown(&context, request, &response);
//     }
//     case Runnable::State::Kill:

//         // Forcibly kill the node process
//         if (m_setup)
//         {
//             ::node::Stop(m_setup->env());
//         }

//         break;

//     default:
//         break;
//     }
// }

// // std::unique_ptr<::node::CommonEnvironmentSetup> NodeRuntime::node_init_setup(std::vector<std::string> args) {}

// void NodeRuntime::launch_node(std::vector<std::string> args)
// {
//     if (!m_init_result)
//     {
//         // Convert the string array to a char**
//         std::vector<char*> raw_args_array = vec_string_to_char_ptr(args);

//         // Initialize uv with the arguments
//         char** argv = uv_setup_args(args.size(), raw_args_array.data());

//         // Update the args (I gues UV can change them?)
//         args = std::vector<std::string>(argv, argv + args.size());

//         // Parse Node.js CLI options, and print any errors that have occurred while
//         // trying to parse them.
//         std::unique_ptr<::node::InitializationResult> result = ::node::InitializeOncePerProcess(
//             args,
//             {::node::ProcessInitializationFlags::kNoInitializeV8,
//              ::node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

//         for (const std::string& error : result->errors())
//         {
//             // fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());
//             LOG(ERROR) << "Error with args. Error: " << error.c_str();
//         }

//         if (static_cast<int>(result->early_return()) != 0)
//         {
//             // return result->exit_code();
//             LOG(ERROR) << "Exited early. Code: " << result->exit_code();
//         }

//         m_init_result = std::move(result);
//     }

//     if (!m_platform)
//     {
//         // Create a v8::Platform instance. `MultiIsolatePlatform::Create()` is a way
//         // to create a v8::Platform instance that Node.js can use when creating
//         // Worker threads. When no `MultiIsolatePlatform` instance is present,
//         // Worker threads are disabled.
//         std::unique_ptr<::node::MultiIsolatePlatform> platform = ::node::MultiIsolatePlatform::Create(4);
//         v8::V8::InitializePlatform(platform.get());
//         v8::V8::Initialize();

//         m_platform = std::move(platform);
//     }

//     if (!m_setup)
//     {
//         std::vector<std::string> errors;
//         std::unique_ptr<::node::CommonEnvironmentSetup> setup = ::node::CommonEnvironmentSetup::Create(
//             m_platform.get(),
//             &errors,
//             m_init_result->args(),
//             m_init_result->exec_args());

//         if (!setup)
//         {
//             LOG(ERROR) << "Error creating setup:";
//             for (const std::string& err : errors)
//             {
//                 // fprintf(stderr, "%s: %s\n", args[0].c_str(), err.c_str());
//                 LOG(ERROR) << err;
//             }
//         }

//         m_setup = std::move(setup);
//     }

//     // Now run the environment
//     node_run_environment(m_setup);

//     // Finally, cleanup
//     v8::V8::Dispose();
//     v8::V8::DisposePlatform();

//     ::node::TearDownOncePerProcess();
// }

NodeService::NodeService(runnable::IRunnableResourcesProvider& resources, std::vector<std::string> args) :
  AsyncService("NodeService"),
  runnable::RunnableResourcesProvider(resources),
  m_args(std::move(args))
{
    auto mrc_lib_location = std::filesystem::path(utils::get_mrc_lib_location());

    auto node_service_js = mrc_lib_location.parent_path() / "mrc" / "control-plane" / "server" / "run_server.mjs";

    m_started_future = m_started_promise.get_future();

    // m_launch_node = std::getenv("MRC_SKIP_LAUNCH_NODE") == nullptr;

    // if (!m_launch_node)
    // {
    //     LOG(INFO) << "Environment variable MRC_SKIP_LAUNCH_NODE was set and the control plane will not be run.";
    // }
}

NodeService::~NodeService()
{
    if (m_node_thread.joinable())
    {
        m_node_thread.join();
    }

    AsyncService::call_in_destructor();
}

// void NodeService::set_args(std::vector<std::string> args)
// {
//     m_args = std::move(args);
// }

void NodeService::do_service_start(std::stop_token stop_token)
{
    boost::fibers::packaged_task<void()> pkg_task(std::move([this, stop_token]() {
        this->launch_node(m_args);
    }));

    m_completed_future = pkg_task.get_future();

    m_node_thread = std::thread(std::move(pkg_task));
}
void NodeService::do_service_kill()
{
    DVLOG(10) << "[Node] do_service_kill() started";

    // Forcibly kill the node process
    if (m_setup)
    {
        ::node::Stop(m_setup->env());
    }

    DVLOG(10) << "[Node] do_service_kill() complete";
}

// void NodeService::do_service_stop()
// {
//     DVLOG(10) << "[Node] do_service_stop() started";

//     if (m_launch_node)
//     {
//         // Send a gRPC message to shutdown the server
//         auto channel = grpc::CreateChannel("localhost:13337", grpc::InsecureChannelCredentials());
//         auto stub    = mrc::protos::Architect::NewStub(channel);

//         auto context = grpc::ClientContext();

//         ::mrc::protos::ShutdownRequest request;
//         ::mrc::protos::ShutdownResponse response;

//         stub->Shutdown(&context, request, &response);
//     }

//     DVLOG(10) << "[Node] do_service_stop() complete";
// }

// void NodeService::do_service_await_live()
// {
//     DVLOG(10) << "[Node] do_service_await_live() started";

//     // Wait for the service to start
//     m_started_future.get();

//     // Now ping the server to check its OK
//     auto channel = grpc::CreateChannel("localhost:13337", grpc::InsecureChannelCredentials());
//     auto stub    = mrc::protos::Architect::NewStub(channel);

//     ::mrc::protos::PingRequest request;
//     ::mrc::protos::PingResponse response;

//     request.set_tag(1235);

//     grpc::Status status;

//     do
//     {
//         auto context = grpc::ClientContext();
//         status       = stub->Ping(&context, request, &response);
//     } while (!status.ok());

//     DVLOG(10) << "Ping response: " << response.tag();

//     DVLOG(10) << "[Node] do_service_await_live() complete";
// }

// void NodeService::do_service_await_join()
// {
//     DVLOG(10) << "[Node] do_service_await_join() started";

//     // Wait for the completed future to be done
//     m_completed_future.get();

//     DVLOG(10) << "[Node] do_service_await_join() complete";
// }

void NodeService::launch_node(std::vector<std::string> args)
{
    DVLOG(10) << "[Node] Launching node with args: " << utils::StringUtil::array_to_str(args.begin(), args.end());

    if (!m_init_result)
    {
        // Convert the string array to a char**
        std::vector<char*> raw_args_array = vec_string_to_char_ptr(args);

        // Initialize uv with the arguments
        char** argv = uv_setup_args(args.size(), raw_args_array.data());

        // Update the args (I gues UV can change them?)
        args = std::vector<std::string>(argv, argv + args.size());

        // Parse Node.js CLI options, and print any errors that have occurred while
        // trying to parse them.
        std::unique_ptr<::node::InitializationResult> result = ::node::InitializeOncePerProcess(
            args,
            {::node::ProcessInitializationFlags::kNoInitializeV8,
             ::node::ProcessInitializationFlags::kNoInitializeNodeV8Platform});

        for (const std::string& error : result->errors())
        {
            // fprintf(stderr, "%s: %s\n", args[0].c_str(), error.c_str());
            LOG(ERROR) << "Error with args. Error: " << error.c_str();
        }

        if (static_cast<int>(result->early_return()) != 0)
        {
            // return result->exit_code();
            LOG(ERROR) << "Exited early. Code: " << result->exit_code();
        }

        m_init_result = std::move(result);
    }

    if (!m_platform)
    {
        // Create a v8::Platform instance. `MultiIsolatePlatform::Create()` is a way
        // to create a v8::Platform instance that Node.js can use when creating
        // Worker threads. When no `MultiIsolatePlatform` instance is present,
        // Worker threads are disabled.
        std::unique_ptr<::node::MultiIsolatePlatform> platform = ::node::MultiIsolatePlatform::Create(4);
        v8::V8::InitializePlatform(platform.get());
        v8::V8::Initialize();

        m_platform = std::move(platform);
    }

    if (!m_setup)
    {
        std::vector<std::string> errors;
        std::unique_ptr<::node::CommonEnvironmentSetup> setup = ::node::CommonEnvironmentSetup::Create(
            m_platform.get(),
            &errors,
            m_init_result->args(),
            m_init_result->exec_args());

        if (!setup)
        {
            LOG(ERROR) << "Error creating setup:";
            for (const std::string& err : errors)
            {
                // fprintf(stderr, "%s: %s\n", args[0].c_str(), err.c_str());
                LOG(ERROR) << err;
            }
        }

        m_setup = std::move(setup);
    }

    // Now run the environment
    int exit_code = 0;

    v8::Isolate* isolate     = m_setup->isolate();
    ::node::Environment* env = m_setup->env();

    {
        v8::Locker locker(isolate);
        v8::Isolate::Scope isolate_scope(isolate);
        v8::HandleScope handle_scope(isolate);
        v8::Context::Scope context_scope(m_setup->context());

        auto loadenv_ret = ::node::LoadEnvironment(env, ::node::StartExecutionCallback{});

        if (loadenv_ret.IsEmpty())
        {
            // There has been a JS exception.
            LOG(ERROR) << "There has been a JS exception.";
            // started_promise.set_exception(std::exception_ptr p);
        }

        DVLOG(10) << "[Node] Node environment started. Beginning Spin loop";

        // Set the value
        m_started_promise.set_value();

        exit_code = ::node::SpinEventLoop(env).FromMaybe(1);

        DVLOG(10) << "[Node] Spin loop complete. Calling Stop";

        ::node::Stop(env);
    }

    // Destroy the environment
    m_setup.reset();

    // Finally, cleanup
    v8::V8::Dispose();
    v8::V8::DisposePlatform();

    ::node::TearDownOncePerProcess();

    DVLOG(10) << "[Node] Node teardown complete";
}

}  // namespace mrc::control_plane
