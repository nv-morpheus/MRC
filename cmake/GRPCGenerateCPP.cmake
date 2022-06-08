# SPDX-FileCopyrightText: Copyright (c) 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# Separate the output files from protobuf_generate into src, headers and descriptors. Taken from PROTOBUF_GENERATE_CPP
function(protobuf_separate_output output_files)

  cmake_parse_arguments(protobuf_separate_output "" "HDRS;SRCS;DESCRIPTORS" "" "${ARGN}")

  set(SRCS "")
  set(HDRS "")
  set(DESCRIPTORS "")

  foreach(_file ${output_files})
    if(_file MATCHES "cc$")
      list(APPEND SRCS ${_file})
    elseif(_file MATCHES "desc$")
      list(APPEND DESCRIPTORS ${_file})
    else()
      list(APPEND HDRS ${_file})
    endif()
  endforeach()

  if (protobuf_separate_output_HDRS)
    list(APPEND ${protobuf_separate_output_HDRS} ${HDRS})
    list(REMOVE_DUPLICATES ${protobuf_separate_output_HDRS})
    set(${protobuf_separate_output_HDRS} ${${protobuf_separate_output_HDRS}} PARENT_SCOPE)
  endif()
  if (protobuf_separate_output_SRCS)
    list(APPEND ${protobuf_separate_output_SRCS} ${SRCS})
    list(REMOVE_DUPLICATES ${protobuf_separate_output_SRCS})
    set(${protobuf_separate_output_SRCS} ${${protobuf_separate_output_SRCS}} PARENT_SCOPE)
  endif()
  if (protobuf_separate_output_DESCRIPTORS)
    list(APPEND ${protobuf_separate_output_DESCRIPTORS} ${DESCRIPTORS})
    list(REMOVE_DUPLICATES ${protobuf_separate_output_DESCRIPTORS})
    set(${protobuf_separate_output_DESCRIPTORS} ${${protobuf_separate_output_DESCRIPTORS}} PARENT_SCOPE)
  endif()

endfunction()

# Generates CPP gRPC services. Use GEN_GRPC to indicate whether or not to use the gRPC extension
function(protobuf_generate_grpc_cpp target)

  cmake_parse_arguments(protobuf_generate_grpc_cpp "GEN_GRPC" "HDRS;SRCS;DESCRIPTORS" "PROTOS" ${ARGN})

  set(out_files)

  # Generate the cpp files
  protobuf_generate(
    LANGUAGE cpp
    OUT_VAR out_files
    IMPORT_DIRS ${Protobuf_IMPORT_DIRS}
    TARGET ${target}
    PROTOS ${protobuf_generate_grpc_cpp_PROTOS}
    ${protobuf_generate_grpc_cpp_UNPARSED_ARGUMENTS}
  )

  protobuf_separate_output(
    "${out_files}"
    HDRS ${protobuf_generate_grpc_cpp_HDRS}
    SRCS ${protobuf_generate_grpc_cpp_SRCS}
    DESCRIPTORS ${protobuf_generate_grpc_cpp_DESCRIPTORS}
  )

  if (protobuf_generate_grpc_cpp_GEN_GRPC)
    set(out_files "")

    # Generate the grpc files
    protobuf_generate(
      LANGUAGE grpc
      OUT_VAR out_files
      GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
      PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
      IMPORT_DIRS ${Protobuf_IMPORT_DIRS}
      TARGET ${target}
      PROTOS ${protobuf_generate_grpc_cpp_PROTOS}
      ${protobuf_generate_grpc_cpp_UNPARSED_ARGUMENTS}
    )

    protobuf_separate_output(
      "${out_files}"
      HDRS ${protobuf_generate_grpc_cpp_HDRS}
      SRCS ${protobuf_generate_grpc_cpp_SRCS}
      DESCRIPTORS ${protobuf_generate_grpc_cpp_DESCRIPTORS}
    )
  endif()

  # Now configure the target common for all proto targets
  target_link_libraries(${target}
    PUBLIC
      protobuf::libprotobuf
  )

  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  )

  # We must always compile protobufs with `NDEBUG` defined due to an issue with
  # libprotobuf>=3.20. Their header files can change between Debug/Release which
  # causes undefined symbol errors when building and running in Debug. Setting
  # this definition gets around this issue by ensuring a consistent value for
  # `NDEBUG`. See this issue for more info:
  # https://github.com/protocolbuffers/protobuf/issues/9947
  target_compile_definitions(${target}
    PRIVATE NDEBUG
  )

  if (protobuf_generate_grpc_cpp_HDRS)
    set(${protobuf_generate_grpc_cpp_HDRS} ${${protobuf_generate_grpc_cpp_HDRS}} PARENT_SCOPE)
    set_target_properties(${target} PROPERTIES PUBLIC_HEADER "${${protobuf_generate_grpc_cpp_HDRS}}")
  endif()

  if (protobuf_generate_grpc_cpp_SRCS)
    set(${protobuf_generate_grpc_cpp_SRCS} ${${protobuf_generate_grpc_cpp_SRCS}} PARENT_SCOPE)
  endif()

endfunction()
