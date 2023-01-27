# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if (VERBOSE)
  morpheus_utils_print_config()
endif()

morpheus_utils_initialize_cpm(MRC_CACHE_DIR)

# Start with CUDA. Need to add it to our export set
rapids_find_package(CUDAToolkit
  REQUIRED
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

# Boost
# =====
morpheus_utils_configure_boost()

# UCX
# ===
morpheus_utils_configure_ucx()

# hwloc
# =====
morpheus_utils_configure_hwloc()

# expected
# ========
morpheus_utils_configure_tl_expected()

# NVIDIA RAPIDS RMM
# =================
morpheus_utils_configure_rmm()

# gflags
# ======
rapids_find_package(gflags REQUIRED
  GLOBAL_TARGETS gflags
  BUILD_EXPORT_SET ${PROJECT_NAME}-core-exports
  INSTALL_EXPORT_SET ${PROJECT_NAME}-core-exports
)

# glog
# ====
# - link against shared
# - todo: compile with -DWITH_GFLAGS=OFF and remove gflags dependency
morpheus_utils_configure_glog()

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
morpheus_utils_configure_rxcpp()

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
morpheus_utils_configure_prometheus_cpp()

# libcudacxx
# =========
morpheus_utils_configure_libcudacxx()

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
