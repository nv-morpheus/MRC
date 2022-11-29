# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

list(APPEND CMAKE_MESSAGE_CONTEXT "dep")

# Initialize rapids CPM with package overrides
rapids_cpm_init(OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/deps/rapids_cpm_package_overrides.json")

# Print CMake settings when verbose output is enabled
message(VERBOSE "PROJECT_NAME: " ${PROJECT_NAME})
message(VERBOSE "CMAKE_HOST_SYSTEM: ${CMAKE_HOST_SYSTEM}")
message(VERBOSE "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})
message(VERBOSE "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(VERBOSE "CMAKE_CXX_COMPILER_ID: " ${CMAKE_CXX_COMPILER_ID})
message(VERBOSE "CMAKE_CXX_COMPILER_VERSION: " ${CMAKE_CXX_COMPILER_VERSION})
message(VERBOSE "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(VERBOSE "CMAKE_CUDA_COMPILER: " ${CMAKE_CUDA_COMPILER})
message(VERBOSE "CMAKE_CUDA_COMPILER_ID: " ${CMAKE_CUDA_COMPILER_ID})
message(VERBOSE "CMAKE_CUDA_COMPILER_VERSION: " ${CMAKE_CUDA_COMPILER_VERSION})
message(VERBOSE "CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
message(VERBOSE "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})
message(VERBOSE "CMAKE_CURRENT_BINARY_DIR: " ${CMAKE_CURRENT_BINARY_DIR})
message(VERBOSE "CMAKE_CURRENT_LIST_DIR: " ${CMAKE_CURRENT_LIST_DIR})
message(VERBOSE "CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS})
message(VERBOSE "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
message(VERBOSE "CMAKE_INSTALL_FULL_INCLUDEDIR: " ${CMAKE_INSTALL_FULL_INCLUDEDIR})
message(VERBOSE "CMAKE_INSTALL_FULL_LIBDIR: " ${CMAKE_INSTALL_FULL_LIBDIR})
message(VERBOSE "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})
message(VERBOSE "CMAKE_PREFIX_PATH: " ${CMAKE_PREFIX_PATH})
message(VERBOSE "CMAKE_FIND_ROOT_PATH: " ${CMAKE_FIND_ROOT_PATH})
message(VERBOSE "CMAKE_LIBRARY_ARCHITECTURE: " ${CMAKE_LIBRARY_ARCHITECTURE})
message(VERBOSE "FIND_LIBRARY_USE_LIB64_PATHS: " ${FIND_LIBRARY_USE_LIB64_PATHS})
message(VERBOSE "CMAKE_SYSROOT: " ${CMAKE_SYSROOT})
message(VERBOSE "CMAKE_STAGING_PREFIX: " ${CMAKE_STAGING_PREFIX})
message(VERBOSE "CMAKE_FIND_ROOT_PATH_MODE_INCLUDE: " ${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE})

# Start with CUDA. Need to add it to our export set
rapids_find_package(CUDAToolkit
  REQUIRED
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

# Boost
# =====
# - Use static linking to avoid issues with system-wide installations of Boost.
# - Use numa=on to ensure the numa component of fiber gets built
set(BOOST_VERSION "1.74.0" CACHE STRING "Version of boost to use")
include(deps/Configure_boost)

# UCX
# ===
set(UCX_VERSION "1.13" CACHE STRING "Version of ucx to use")
include(deps/Configure_ucx)

# hwloc
# =====
set(HWLOC_VERSION "2.5" CACHE STRING "Version of hwloc to use")
include(deps/Configure_hwloc)

# expected
set(EXPECTED_VERSION "1.0.0" CACHE STRING "Version of expected to use")
include(deps/Configure_expected)

# FlatBuffers
# ===========
# rapids_find_package(Flatbuffers REQUIRED
# GLOBAL_TARGETS Flatbuffers
# BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
# INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
# FIND_ARGS
# CONFIG
# )

# NVIDIA RAPIDS RMM
# =================
set(RMM_VERSION "\${MRC_RAPIDS_VERSION}" CACHE STRING "Version of RMM to use. Defaults to \${MRC_RAPIDS_VERSION}")
include(deps/Configure_RMM)

# gflags
# ======
rapids_find_package(gflags REQUIRED
  GLOBAL_TARGETS gflags
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports

  # FIND_ARGS
  # CONFIG
)

# glog
# ====
# - link against shared
# - todo: compile with -DWITH_GFLAGS=OFF and remove gflags dependency
set(GLOG_VERSION "0.6" CACHE STRING "Version of glog to use")
include(deps/Configure_glog)

# nvidia cub
# ==========
find_path(CUB_INCLUDE_DIRS "cub/cub.cuh"
  HINTS ${CUDAToolkit_INCLUDE_DIRS} ${Thrust_SOURCE_DIR} ${CUB_DIR}
  REQUIRE
)

# grpc-repo
# =========
rapids_find_package(gRPC REQUIRED
  GLOBAL_TARGETS
  gRPC::address_sorting gRPC::gpr gRPC::grpc gRPC::grpc_unsecure gRPC::grpc++ gRPC::grpc++_alts gRPC::grpc++_error_details gRPC::grpc++_reflection
  gRPC::grpc++_unsecure gRPC::grpc_plugin_support gRPC::grpcpp_channelz gRPC::upb gRPC::grpc_cpp_plugin gRPC::grpc_csharp_plugin gRPC::grpc_node_plugin
  gRPC::grpc_objective_c_plugin gRPC::grpc_php_plugin gRPC::grpc_python_plugin gRPC::grpc_ruby_plugin
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

# RxCpp
# =====
set(RXCPP_VERSION "4.1.1.2" CACHE STRING "Version of RxCpp to use")
include(deps/Configure_rxcpp)

# JSON
# ======
rapids_find_package(nlohmann_json REQUIRED
  GLOBAL_TARGETS nlohmann_json::nlohmann_json
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
  FIND_ARGS
  CONFIG
)

# prometheus
# =========
set(PROMETHEUS_CPP_VERSION "1.0.0" CACHE STRING "Version of Prometheus-cpp to use")
include(deps/Configure_prometheus)

# libcudacxx
# =========
set(LIBCUDACXX_VERSION "1.8.0" CACHE STRING "Version of libcudacxx to use")
include(deps/Configure_libcudacxx)

if(MRC_BUILD_BENCHMARKS)
  # google benchmark
  # ================
  rapids_find_package(benchmark REQUIRED
    GLOBAL_TARGETS benchmark::benchmark
    BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports

    # No install set
    FIND_ARGS
    CONFIG
  )
endif()

if(MRC_BUILD_TESTS)
  # google test
  # ===========
  rapids_find_package(GTest REQUIRED
    GLOBAL_TARGETS GTest::gtest GTest::gmock GTest::gtest_main GTest::gmock_main
    BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports

    # No install set
    FIND_ARGS
    CONFIG
  )
endif()

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
