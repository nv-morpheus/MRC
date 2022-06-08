# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


####################################
# - Begin SRF Targets --------------

# Keep all source files sorted
add_library(libsrf
  src/public/benchmarking/trace_statistics.cpp
  src/public/benchmarking/tracer.cpp
  src/public/benchmarking/util.cpp
  src/public/channel/channel.cpp
  src/public/codable/encoded_object.cpp
  src/public/core/addresses.cpp
  src/public/core/bitmap.cpp
  src/public/core/executor.cpp
  src/public/core/fiber_pool.cpp
  src/public/core/logging.cpp
  src/public/cuda/device_guard.cpp
  src/public/cuda/sync.cpp
  src/public/manifold/manifold.cpp
  src/public/metrics/counter.cpp
  src/public/metrics/registry.cpp
  src/public/memory/blob.cpp
  src/public/memory/block.cpp
  src/public/node/edge_registry.cpp
  src/public/options/engine_groups.cpp
  src/public/options/fiber_pool.cpp
  src/public/options/options.cpp
  src/public/options/placement.cpp
  src/public/options/resources.cpp
  src/public/options/services.cpp
  src/public/options/topology.cpp
  src/public/runnable/context.cpp
  src/public/runnable/launcher.cpp
  src/public/runnable/runnable.cpp
  src/public/runnable/runner.cpp
  src/public/segment/definition.cpp
  src/public/segment/builder.cpp
# src/public/segment/runnable.cpp
  src/public/runnable/types.cpp
  src/public/utils/bytes_to_string.cpp
  src/public/utils/thread_utils.cpp
  src/public/utils/type_utils.cpp

# src/internal/data_plane/client_worker.cpp
# src/internal/data_plane/client.cpp
# src/internal/data_plane/instance.cpp
# src/internal/data_plane/server.cpp
  src/internal/executor/executor.cpp
  src/internal/executor/iexecutor.cpp
  src/internal/pipeline/controller.cpp
  src/internal/pipeline/instance.cpp
  src/internal/pipeline/pipeline.cpp
  src/internal/pipeline/ipipeline.cpp
  src/internal/pipeline/manager.cpp
  src/internal/pipeline/resources.cpp
  src/internal/resources/device_resources.cpp
  src/internal/resources/host_resources.cpp
  src/internal/resources/partition_resources.cpp
  src/internal/resources/resource_partitions.cpp
  src/internal/resources/system_resources.cpp
  src/internal/runnable/engine_factory.cpp
  src/internal/runnable/engine.cpp
  src/internal/runnable/engines.cpp
  src/internal/runnable/fiber_engine.cpp
  src/internal/runnable/fiber_engines.cpp
  src/internal/runnable/thread_engine.cpp
  src/internal/runnable/thread_engines.cpp
  src/internal/segment/builder.cpp
  src/internal/segment/definition.cpp
  src/internal/segment/idefinition.cpp
  src/internal/segment/instance.cpp
  src/internal/service.cpp
  src/internal/system/device_info.cpp
  src/internal/system/device_partition.cpp
  src/internal/system/engine_factory_cpu_sets.cpp
  src/internal/system/fiber_manager.cpp
  src/internal/system/fiber_pool.cpp
  src/internal/system/fiber_task_queue.cpp
  src/internal/system/gpu_info.cpp
  src/internal/system/host_partition.cpp
  src/internal/system/isystem.cpp
  src/internal/system/partitions.cpp
  src/internal/system/system.cpp
  src/internal/system/thread_pool.cpp
  src/internal/system/topology.cpp
  src/internal/ucx/context.cpp
  src/internal/ucx/endpoint.cpp
  src/internal/ucx/receive_manager.cpp
  src/internal/ucx/worker.cpp
  src/internal/utils/collision_detector.cpp
  src/internal/utils/exception_guard.cpp
  src/internal/utils/parse_config.cpp
  src/internal/utils/parse_ints.cpp
  src/internal/utils/shared_resource_bit_map.cpp
)

add_library(${PROJECT_NAME}::libsrf ALIAS libsrf)

target_link_libraries(libsrf
  PUBLIC
    srf_protos
    srf_architect_protos
    rmm::rmm
    CUDA::nvml
    CUDA::cudart
    rxcpp::rxcpp
    glog::glog
    libcudacxx::libcudacxx
  PRIVATE
    nvrpc
    nvrpc-client
    hwloc::hwloc
    prometheus-cpp::core # private in MR !199
    ucx::ucs
    ucx::ucp
)

target_include_directories(libsrf
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(libsrf
  PUBLIC
    $<$<BOOL:${SRF_BUILD_BENCHMARKS}>:SRF_ENABLE_BENCHMARKING>
)

set_target_properties(libsrf PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

# Finally, set the install RPATH to include the stubs folder for CUDA::nvml. If thats made private, this can be removed
set_target_properties(libsrf PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib/stubs")

###################################################################################################
# - install targets -------------------------------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

install(
  TARGETS libsrf
  DESTINATION ${lib_dir}
  EXPORT ${PROJECT_NAME}-core-exports
  COMPONENT Core
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Core
)


if (SRF_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(src/tests)
endif ()
