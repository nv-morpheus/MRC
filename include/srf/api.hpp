/**
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Generic helper definitions for shared library support from
// https://gcc.gnu.org/wiki/Visibility

// For every non-templated non-static function definition in your library (both headers and source files), decide if it
// is publicly used or internally used: If it is publicly used, mark with SRF_API like this: extern SRF_API PublicFunc()

// If it is only internally used, mark with SRF_LOCAL like this: extern SRF_LOCAL PublicFunc() Remember, static
// functions need no demarcation, nor does anything in an anonymous namespace, nor does anything which is templated.

// For every non-templated class definition in your library (both headers and source files), decide if it is publicly
// used or internally used: If it is publicly used, mark with SRF_API like this: class SRF_API PublicClass

// If it is only internally used, mark with SRF_LOCAL like this: class SRF_LOCAL PublicClass

// Individual member functions of an exported class that are not part of the interface, in particular ones which are
// private, and are not used by friend code, should be marked individually with SRF_LOCAL.

// In your build system (Makefile etc), you will probably wish to add the -fvisibility=hidden and
// -fvisibility-inlines-hidden options to the command line arguments of every GCC invocation. Remember to test your
// library thoroughly afterwards, including that all exceptions correctly traverse shared object boundaries.

#if defined _WIN32 || defined __CYGWIN__
    #define SRF_HELPER_DLL_IMPORT __declspec(dllimport)
    #define SRF_HELPER_DLL_EXPORT __declspec(dllexport)
    #define SRF_HELPER_DLL_LOCAL
#else
    #if __GNUC__ >= 4
        #define SRF_HELPER_DLL_IMPORT __attribute__((visibility("default")))
        #define SRF_HELPER_DLL_EXPORT __attribute__((visibility("default")))
        #define SRF_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
    #else
        #define SRF_HELPER_DLL_IMPORT
        #define SRF_HELPER_DLL_EXPORT
        #define SRF_HELPER_DLL_LOCAL
    #endif
#endif

// Now we use the generic helper definitions above to define SRF_API and SRF_LOCAL.
// SRF_API is used for the public API symbols. It either DLL imports or DLL exports (or does nothing for static build)
// SRF_LOCAL is used for non-api symbols.

#define SRF_DLL  // we alway build the .so/.dll
#ifdef libsrf_EXPORTS
    #define SRF_DLL_EXPORTS
#endif

#ifdef SRF_DLL              // defined if SRF is compiled as a DLL
    #ifdef SRF_DLL_EXPORTS  // defined if we are building the SRF DLL (instead of using it)
        #define SRF_API SRF_HELPER_DLL_EXPORT
    #else
        #define SRF_API SRF_HELPER_DLL_IMPORT
    #endif  // SRF_DLL_EXPORTS
    #define SRF_LOCAL SRF_HELPER_DLL_LOCAL
#else  // SRF_DLL is not defined: this means SRF is a static lib.
    #define SRF_API
    #define SRF_LOCAL
static_assert(false, "always build the .so/.dll")
#endif  // SRF_DLL
