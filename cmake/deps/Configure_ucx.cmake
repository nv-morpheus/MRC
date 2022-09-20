#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

function(find_and_configure_ucx version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "ucx")

  # Try to find UCX and download from source if not found
  rapids_cpm_find(ucx 1.12
    GLOBAL_TARGETS
      ucx ucx::ucp ucx::uct ucx_ucx ucx::ucp ucx::uct ucx::ucx
    BUILD_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    INSTALL_EXPORT_SET
      ${PROJECT_NAME}-core-exports
    CPM_ARGS
      GIT_REPOSITORY          https://github.com/openucx/ucx.git
      GIT_TAG                 "v${version}"
      DOWNLOAD_ONLY           TRUE
  )

  if (ucx_ADDED)
    # If we got here, UCX wasnt found. Add custom targets to build from source.
    message(STATUS "UCX Not Found. Building from source.")

    # Location where all files will be temp installed
    set(ucx_INSTALL_DIR ${ucx_BINARY_DIR}/install)

    # Because libtool shows an error when calling `make install
    # prefix=<INSTALL_DIR>`. We have to use DESTDIR instead. Make a symbolic
    # link to prevent very long names in the include paths
    set(ucx_DEST_DIR ${ucx_BINARY_DIR}/install_dest)

    # Ensure the output exists
    file(MAKE_DIRECTORY ${ucx_DEST_DIR}/${CMAKE_INSTALL_PREFIX})
    file(CREATE_LINK ${ucx_DEST_DIR}/${CMAKE_INSTALL_PREFIX} ${ucx_INSTALL_DIR} SYMBOLIC)
    file(MAKE_DIRECTORY ${ucx_BINARY_DIR}/install)
    file(MAKE_DIRECTORY ${ucx_INSTALL_DIR}/include)

    # file(MAKE_DIRECTORY ${ucx_BINARY_DIR}/src)

    include(ExternalProject)

    string(TOUPPER ${CMAKE_BUILD_TYPE} BUILD_TYPE_UC)

    # Get the Compiler settings to forward onto autoconf
    set(COMPILER_SETTINGS
      "CXX=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_CXX_COMPILER}"
      "CPP=${CMAKE_CXX_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER} -E"
      "CC=${CMAKE_C_COMPILER_LAUNCHER} ${CMAKE_C_COMPILER}"
      "AR=${CMAKE_C_COMPILER_AR}"
      "RANLIB=${CMAKE_C_COMPILER_RANLIB}"
      "NM=${CMAKE_NM}"
      "STRIP=${CMAKE_STRIP}"
      "CFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}}"
      "CPPFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}}"
      "CXXFLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UC}}"
      "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE_UC}}"
    )

    # Use BUILD_IN_SOURCE because UCX cant do out of source builds and CMake sucks at change directory
    ExternalProject_Add(ucx
      PREFIX            ${ucx_BINARY_DIR}
      SOURCE_DIR        ${ucx_BINARY_DIR}
      INSTALL_DIR       ${ucx_INSTALL_DIR}
      # Copy from CPM cache into the build tree
      DOWNLOAD_COMMAND  ${CMAKE_COMMAND} -E copy_directory ${ucx_SOURCE_DIR} ${ucx_BINARY_DIR}
      # The io_demo fails to build in out of source builds. So remove that from
      # the Makefile (wish we could just disable all test/examples/apps) as a
      # part of the download command
      PATCH_COMMAND     git checkout -- . && git apply --whitespace=fix ${PROJECT_SOURCE_DIR}/cmake/deps/patches/ucx.patch
      # Note, we set SED and GREP here since they can be hard coded in the conda libtoolize
      CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env SED=sed GREP=grep <SOURCE_DIR>/autogen.sh
                COMMAND <SOURCE_DIR>/contrib/configure-release ${COMPILER_SETTINGS} --prefix=${CMAKE_INSTALL_PREFIX} --enable-mt --without-rdmacm --disable-gtest --disable-examples
      BUILD_COMMAND     make -j
      BUILD_IN_SOURCE   TRUE
      BUILD_BYPRODUCTS  <INSTALL_DIR>/lib/libuct.a
                        <INSTALL_DIR>/lib/libucp.a
                        <INSTALL_DIR>/lib/libucs.a
                        <INSTALL_DIR>/lib/libucm.a
      INSTALL_COMMAND   make DESTDIR=${ucx_DEST_DIR} install
      LOG_CONFIGURE     TRUE
      LOG_BUILD         TRUE
      LOG_INSTALL       TRUE
      # Add a target for configuring to allow for style checks on source code
      STEP_TARGETS      install
    )

    # Install only the headers
    install(
      DIRECTORY ${ucx_INSTALL_DIR}/include
      TYPE INCLUDE
    )

    # TODO: We support components in the custom build branch but not when UCX is found via find_package.
    # UCT Library
    add_library(ucx::uct STATIC IMPORTED GLOBAL)
    set_target_properties(ucx::uct PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${ucx_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
      INTERFACE_POSITION_INDEPENDENT_CODE "ON"
    )
    set_property(TARGET ucx::uct APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(ucx::uct PROPERTIES
      IMPORTED_LOCATION_RELEASE "${ucx_INSTALL_DIR}/lib/libuct.a"
      IMPORTED_SONAME_RELEASE "libuct.a"
    )
    add_dependencies(ucx::uct ucx)

    # UCP Library
    add_library(ucx::ucp STATIC IMPORTED GLOBAL)
    set_target_properties(ucx::ucp PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${ucx_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
      INTERFACE_POSITION_INDEPENDENT_CODE "ON"
    )
    set_property(TARGET ucx::ucp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(ucx::ucp PROPERTIES
      IMPORTED_LOCATION_RELEASE "${ucx_INSTALL_DIR}/lib/libucp.a"
      IMPORTED_SONAME_RELEASE "libucp.a"
    )
    add_dependencies(ucx::ucp ucx)

    # UCS Library
    add_library(ucx::ucs STATIC IMPORTED GLOBAL)
    set_target_properties(ucx::ucs PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${ucx_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
      INTERFACE_POSITION_INDEPENDENT_CODE "ON"
    )
    set_property(TARGET ucx::ucs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(ucx::ucs PROPERTIES
      IMPORTED_LOCATION_RELEASE "${ucx_INSTALL_DIR}/lib/libucs.a"
      IMPORTED_SONAME_RELEASE "libucs.a"
    )
    add_dependencies(ucx::ucs ucx)

    # UCM Library
    add_library(ucx::ucm STATIC IMPORTED GLOBAL)
    set_target_properties(ucx::ucm PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${ucx_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
      INTERFACE_POSITION_INDEPENDENT_CODE "ON"
    )
    set_property(TARGET ucx::ucm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
    set_target_properties(ucx::ucm PROPERTIES
      IMPORTED_LOCATION_RELEASE "${ucx_INSTALL_DIR}/lib/libucm.a"
      IMPORTED_SONAME_RELEASE "libucm.a"
    )
    add_dependencies(ucx::ucm ucx)

    # Combined ucx::ucx target
    add_library(ucx::ucx INTERFACE IMPORTED)
    set_target_properties(ucx::ucx PROPERTIES
      INTERFACE_LINK_LIBRARIES "ucx::uct;ucx::ucp;ucx::ucs;ucx::ucm"
    )

    # Finally, add this to the style check dependencies
    add_dependencies(${PROJECT_NAME}_style_checks ucx-install)
  else()
    # Found installed UCX. Make sure to call rapids_export_package without a version.
    # Otherwise CMake fails with trying to add the dependency twice
    rapids_export_package(
      INSTALL ucx ${PROJECT_NAME}-core-exports
      GLOBAL_TARGETS ucx ucx::ucp ucx::uct ucx_ucx ucx::ucp ucx::uct ucx::ucx
    )

  endif()

endfunction()

find_and_configure_ucx(${UCX_VERSION})
