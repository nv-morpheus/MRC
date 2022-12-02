# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ##################################################################################################
# - libmrc -----------------------------------------------------------------------------------------

# Keep all source files sorted!!!
add_library(libmrc
  src/internal/codable/codable_storage.cpp
  src/internal/codable/decodable_storage_view.cpp
  src/internal/codable/storage_view.cpp
  src/internal/control_plane/client.cpp
  src/internal/control_plane/client/connections_manager.cpp
  src/internal/control_plane/client/instance.cpp
  src/internal/control_plane/client/state_manager.cpp
  src/internal/control_plane/client/subscription_service.cpp
  src/internal/control_plane/resources.cpp
  src/internal/control_plane/server.cpp
  src/internal/control_plane/server/connection_manager.cpp
  src/internal/control_plane/server/subscription_manager.cpp
  src/internal/control_plane/server/tagged_issuer.cpp
  src/internal/data_plane/callbacks.cpp
  src/internal/data_plane/client.cpp
  src/internal/data_plane/request.cpp
  src/internal/data_plane/resources.cpp
  src/internal/data_plane/server.cpp
  src/internal/executor/executor.cpp
  src/internal/executor/iexecutor.cpp
  src/internal/grpc/progress_engine.cpp
  src/internal/grpc/server.cpp
  src/internal/memory/device_resources.cpp
  src/internal/memory/host_resources.cpp
  src/internal/memory/transient_pool.cpp
  src/internal/network/resources.cpp
  src/internal/pipeline/controller.cpp
  src/internal/pipeline/instance.cpp
  src/internal/pipeline/ipipeline.cpp
  src/internal/pipeline/manager.cpp
  src/internal/pipeline/pipeline.cpp
  src/internal/pipeline/port_graph.cpp
  src/internal/pipeline/resources.cpp
  src/internal/pubsub/publisher_round_robin.cpp
  src/internal/pubsub/publisher_service.cpp
  src/internal/pubsub/subscriber_service.cpp
  src/internal/remote_descriptor/decodable_storage.cpp
  src/internal/remote_descriptor/manager.cpp
  src/internal/remote_descriptor/remote_descriptor.cpp
  src/internal/remote_descriptor/storage.cpp
  src/internal/resources/manager.cpp
  src/internal/resources/partition_resources_base.cpp
  src/internal/resources/partition_resources.cpp
  src/internal/runnable/engine_factory.cpp
  src/internal/runnable/engine.cpp
  src/internal/runnable/engines.cpp
  src/internal/runnable/fiber_engine.cpp
  src/internal/runnable/fiber_engines.cpp
  src/internal/runnable/resources.cpp
  src/internal/runnable/thread_engine.cpp
  src/internal/runnable/thread_engines.cpp
  src/internal/runtime/partition.cpp
  src/internal/runtime/runtime.cpp
  src/internal/segment/builder.cpp
  src/internal/segment/definition.cpp
  src/internal/segment/ibuilder.cpp
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
  src/internal/system/host_partition_provider.cpp
  src/internal/system/host_partition.cpp
  src/internal/system/iresources.cpp
  src/internal/system/isystem.cpp
  src/internal/system/partition_provider.cpp
  src/internal/system/partition.cpp
  src/internal/system/partitions.cpp
  src/internal/system/resources.cpp
  src/internal/system/system_provider.cpp
  src/internal/system/system.cpp
  src/internal/system/thread_pool.cpp
  src/internal/system/thread.cpp
  src/internal/system/topology.cpp
  src/internal/ucx/context.cpp
  src/internal/ucx/endpoint.cpp
  src/internal/ucx/memory_block.cpp
  src/internal/ucx/receive_manager.cpp
  src/internal/ucx/resources.cpp
  src/internal/ucx/worker.cpp
  src/internal/utils/collision_detector.cpp
  src/internal/utils/exception_guard.cpp
  src/internal/utils/parse_config.cpp
  src/internal/utils/parse_ints.cpp
  src/internal/utils/shared_resource_bit_map.cpp
  src/public/benchmarking/trace_statistics.cpp
  src/public/benchmarking/tracer.cpp
  src/public/benchmarking/util.cpp
  src/public/channel/channel.cpp
  src/public/codable/encoded_object.cpp
  src/public/codable/memory.cpp
  src/public/core/addresses.cpp
  src/public/core/bitmap.cpp
  src/public/core/executor.cpp
  src/public/core/fiber_pool.cpp
  src/public/core/logging.cpp
  src/public/cuda/device_guard.cpp
  src/public/cuda/sync.cpp
  src/public/manifold/manifold.cpp
  src/public/memory/buffer_view.cpp
  src/public/memory/codable/buffer.cpp
  src/public/metrics/counter.cpp
  src/public/metrics/registry.cpp
  src/public/modules/module_registry.cpp
  src/public/modules/plugins.cpp
  src/public/modules/sample_modules.cpp
  src/public/modules/segment_modules.cpp
  src/public/node/edge_adapter_registry.cpp
  src/public/node/edge_builder.cpp
  src/public/node/edge_registry.cpp
  src/public/node/port_registry.cpp
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
  src/public/runnable/types.cpp
  src/public/runtime/remote_descriptor.cpp
  src/public/segment/builder.cpp
  src/public/segment/definition.cpp
  src/public/utils/bytes_to_string.cpp
  src/public/utils/thread_utils.cpp
  src/public/utils/type_utils.cpp
)

add_library(${PROJECT_NAME}::libmrc ALIAS libmrc)

target_link_libraries(libmrc
  PUBLIC
    mrc_protos
    mrc_architect_protos
    rmm::rmm
    CUDA::nvml
    CUDA::cudart
    rxcpp::rxcpp
    glog::glog
    libcudacxx::libcudacxx
    Boost::fiber
    Boost::context
    glog::glog
    gRPC::grpc++
    gRPC::grpc
    gRPC::gpr
  PRIVATE
    hwloc::hwloc
    prometheus-cpp::core # private in MR !199
    ucx::ucs
    ucx::ucp
    tl::expected
)

target_include_directories(libmrc
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/autogenerated/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(libmrc
  PUBLIC
    $<$<BOOL:${MRC_BUILD_BENCHMARKS}>:MRC_ENABLE_BENCHMARKING>
)

set_target_properties(libmrc PROPERTIES OUTPUT_NAME ${PROJECT_NAME})

# Finally, set the install RPATH to include the stubs folder for CUDA::nvml. If thats made private, this can be removed
set_target_properties(libmrc PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN:${CMAKE_INSTALL_PREFIX}/lib/stubs")

# ##################################################################################################
# - install targets --------------------------------------------------------------------------------

rapids_cmake_install_lib_dir(lib_dir)
include(CPack)
include(GNUInstallDirs)

install(
  TARGETS libmrc
  DESTINATION ${lib_dir}
  EXPORT ${PROJECT_NAME}-core-exports
  COMPONENT Core
)

install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Core
)

# ##################################################################################################
# - subdirectories ---------------------------------------------------------------------------------

if(MRC_BUILD_TESTS)
  add_subdirectory(tests)

  add_subdirectory(src/tests)
endif()

if(MRC_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ##################################################################################################
# - install export ---------------------------------------------------------------------------------

set(doc_string
  [=[
Provide targets for mrc.
]=])

set(code_string "")

set(rapids_project_version_compat SameMinorVersion)

# Need to explicitly set VERSION ${PROJECT_VERSION} here since rapids_cmake gets
# confused with the `RAPIDS_VERSION` variable we use
rapids_export(INSTALL ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libmrc
  VERSION ${PROJECT_VERSION}
  NAMESPACE mrc::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# - build export ----------------------------------------------------------------------------------
rapids_export(BUILD ${PROJECT_NAME}
  EXPORT_SET ${PROJECT_NAME}-core-exports
  GLOBAL_TARGETS libmrc
  VERSION ${PROJECT_VERSION}
  LANGUAGES C CXX CUDA
  NAMESPACE mrc::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)
