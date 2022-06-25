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

#include "./thread_utils.hpp"

#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32

void set_thread_name(uint32_t dwThreadID, const std::string& thread_name)
{
    // DWORD dwThreadID = ::GetThreadId( static_cast<HANDLE>( t.native_handle() ) );

    THREADNAME_INFO info;
    info.dwType     = 0x1000;
    info.szName     = thread_name.c_str();
    info.dwThreadID = dwThreadID;
    info.dwFlags    = 0;

    __try
    {
        RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
    } __except (EXCEPTION_EXECUTE_HANDLER)
    {}
}

void set_thread_name(const std::string& thread_name)
{
    set_thread_name(GetCurrentThreadId(), thread_name);
}

void set_thread_name(const std::thread& t, const std::string& thread_name)
{
    DWORD threadId = ::GetThreadId(static_cast<HANDLE>(thread.native_handle()));
    set_thread_name(threadId, thread_name);
}

#elif defined(__linux__)
    #include <sys/prctl.h>
void set_current_thread_name(const std::string& thread_name)
{
    prctl(PR_SET_NAME, thread_name.c_str(), 0, 0, 0);
}
void set_thread_name(const std::thread& t, const std::string& thread_name)
{
    throw std::logic_error("Function not yet implemented");
}
#else
void set_current_thread_name(const std::string& thread_name)
{
    // TODO(MDD): Figure out how to get this_thread native handle
    auto handle = thread->native_handle();
    pthread_setname_np(handle, thread_name.c_str());
}

void set_thread_name(std::thread* thread, const std::string& thread_name)
{
    auto handle = thread->native_handle();
    pthread_setname_np(handle, thread_name.c_str());
}
#endif
