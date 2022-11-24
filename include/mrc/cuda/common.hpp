/**
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <glog/logging.h>

#include <ostream>  // for logging
#include <string>

// clang-format off

// Global setting to check for CUDA errors. Can be overridden by the build.
// When enabled, will check for errors in Debug build, and will not check for errors in Release
#ifndef MRC_CUDA_CHECK_ERRORS
    #define MRC_CUDA_CHECK_ERRORS 1
#endif

// Forces synchronizing CUDA and checking for error even in a Release build. Has no effect if MRC_CUDA_CHECK_ERRORS=0
#ifndef MRC_CUDA_FORCE_CHECK_ERRORS
    #define MRC_CUDA_FORCE_CHECK_ERRORS 0
#endif

// clang-format on

// Different error messages depending on whether driver or runtime files were included
#ifdef __CUDA_RUNTIME_API_H__
// CUDA Runtime error messages
static std::string __cuda_get_error_string(cudaError_t error)  // NOLINT(readability-identifier-naming)
{
    return std::string(cudaGetErrorString(error));
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static std::string __cuda_get_error_string(CUresult error)
{
    std::stringstream errMsg;
    errMsg << error;
    return errMsg.str();
}
#endif

// This function allows setting breakpoints in code if there is a failure
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
void __check_cuda_errors(T status, const std::string& methodName, const std::string& filename, const int& lineNumber)
{
    if (status)
    {
        LOG(ERROR) << "CUDA Error: " << methodName << " failed. Error message: " << __cuda_get_error_string(status)
                   << "\n"
                   << filename << "(" << lineNumber << ")";
    }
}

#if (!defined(NDEBUG) || MRC_CUDA_FORCE_CHECK_ERRORS) && MRC_CUDA_CHECK_ERRORS

    // Checks the return value of a cuda function
    #define MRC_CHECK_CUDA(code)                                  \
        {                                                         \
            __check_cuda_errors(code, #code, __FILE__, __LINE__); \
        }

    // Syncronizes the current device and checks the last error via cudaGetLastError()
    #define MRC_CHECK_CUDA_LAST_ERROR()        \
        {                                      \
            cudaThreadSynchronize();           \
            MRC_CHECK_CUDA(cudaGetLastError()) \
        }

#else

    // Checks the return value of a cuda function
    #define MRC_CHECK_CUDA(code) code;

    // Syncronizes the current device and checks the last error via cudaGetLastError()
    #define MRC_CHECK_CUDA_LAST_ERROR()

#endif
