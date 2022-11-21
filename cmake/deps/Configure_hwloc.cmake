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

find_package(CUDAToolkit)

function(find_and_configure_hwloc version)

  list(APPEND CMAKE_MESSAGE_CONTEXT "hwloc")

  set(oneValueArgs VERSION PINNED_TAG)
  cmake_parse_arguments(PKG "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN} )

  # # Generate the find modules since hwloc doesnt do this for us
  # rapids_find_generate_module(hwloc
  #   HEADER_NAMES
  #     hwloc.h
  #   LIBRARY_NAMES
  #     hwloc
  #   VERSION
  #     hwloc_VERSION
  # )

  find_package(PkgConfig)

  pkg_check_modules(hwloc IMPORTED_TARGET GLOBAL hwloc)

  if (hwloc_FOUND)

    message(STATUS "Found hwloc with pkg-config at: ${hwloc_LIBRARY_DIRS}")

    # Add an alias to the imported target
    add_library(hwloc::hwloc ALIAS PkgConfig::hwloc)

    set(name "hwloc")

    # Now add it to the list of packages to install
    rapids_export_package(INSTALL hwloc
      ${PROJECT_NAME}-core-exports
      GLOBAL_TARGETS hwloc::hwloc
    )

    # Overwrite the default package contents
    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/templates/pkgconfig_package.cmake.in"
      "${CMAKE_BINARY_DIR}/rapids-cmake/${PROJECT_NAME}-core-exports/install/package_hwloc.cmake" @ONLY)

  else()

    # Try to find hwloc and download from source if not found
    rapids_cpm_find(hwloc ${version}
      GLOBAL_TARGETS
        hwloc hwloc::hwloc
      BUILD_EXPORT_SET
        ${PROJECT_NAME}-core-exports
      INSTALL_EXPORT_SET
        ${PROJECT_NAME}-core-exports
      CPM_ARGS
        GIT_REPOSITORY          https://github.com/open-mpi/hwloc.git
        GIT_TAG                 "hwloc-${version}"
        DOWNLOAD_ONLY           TRUE
        FIND_PACKAGE_ARGUMENTS  "EXACT"
    )

    if (hwloc_ADDED)
      # If we got here, hwloc wasnt found. Add custom targets to build from source.
      message(STATUS "hwloc not installed. Building from source.")

      # Location where all files will be temp installed
      set(hwloc_INSTALL_DIR ${hwloc_BINARY_DIR}/install)

      # Ensure the output exists
      # file(MAKE_DIRECTORY ${hwloc_INSTALL_DIR})
      file(MAKE_DIRECTORY ${hwloc_INSTALL_DIR}/include)
      # file(MAKE_DIRECTORY ${hwloc_BINARY_DIR}/include)

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
        "CPPFLAGS=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UC}} -I${CUDAToolkit_INCLUDE_DIRS}" # Add CUDAToolkit here
        "CXXFLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UC}}"
        "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${BUILD_TYPE_UC}}"
      )

      ExternalProject_Add(hwloc
        PREFIX            ${hwloc_BINARY_DIR}
        SOURCE_DIR        ${hwloc_BINARY_DIR}
        INSTALL_DIR       ${hwloc_INSTALL_DIR}
        # Copy from CPM cache into the build tree
        DOWNLOAD_COMMAND  ${CMAKE_COMMAND} -E copy_directory ${hwloc_SOURCE_DIR} ${hwloc_BINARY_DIR}
        # Note, we set SED and GREP here since they can be hard coded in the conda libtoolize
        CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env SED=sed GREP=grep <SOURCE_DIR>/autogen.sh
                  COMMAND <SOURCE_DIR>/configure ${COMPILER_SETTINGS} --prefix=${CMAKE_INSTALL_PREFIX} --enable-plugins=linux,x86,nvml,pcie,xml --enable-static
        BUILD_COMMAND     make -j
        BUILD_IN_SOURCE   TRUE
        BUILD_BYPRODUCTS  <INSTALL_DIR>/lib/libhwloc.a
        INSTALL_COMMAND   make install prefix=<INSTALL_DIR>
        LOG_CONFIGURE     TRUE
        LOG_BUILD         TRUE
        LOG_INSTALL       TRUE
        # Add a target for configuring to allow for style checks on source code
        STEP_TARGETS      install
      )

      # Install only the headers
      install(
        DIRECTORY ${hwloc_INSTALL_DIR}/include
        TYPE INCLUDE
      )

      add_library(hwloc::hwloc STATIC IMPORTED GLOBAL)

      set_target_properties(hwloc::hwloc PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${hwloc_INSTALL_DIR}/include>;$<INSTALL_INTERFACE:include>"
        INTERFACE_POSITION_INDEPENDENT_CODE "ON"
      )
      set_property(TARGET hwloc::hwloc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(hwloc::hwloc PROPERTIES
        IMPORTED_LOCATION_RELEASE "${hwloc_INSTALL_DIR}/lib/libhwloc.a"
        IMPORTED_SONAME_RELEASE "libhwloc.a"
      )

      # # Add public dependency of xml2
      # set_target_properties(hwloc::hwloc PROPERTIES
      #   INTERFACE_LINK_LIBRARIES "xml2"
      # )
      add_dependencies(hwloc::hwloc hwloc)

      # Finally, add this to the style check dependencies
      add_dependencies(${PROJECT_NAME}_style_checks hwloc-install)

    endif()
  endif()

endfunction()

find_and_configure_hwloc(${HWLOC_VERSION})
