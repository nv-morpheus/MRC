# =============================================================================
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

## TODO: these need to be extracted to a cmake utilities repo

# Ensure we only include this once
include_guard(DIRECTORY)

# Get the project name in uppercase if OPTION_PREFIX is not defined
if(NOT DEFINED OPTION_PREFIX)
  string(TOUPPER "${PROJECT_NAME}" OPTION_PREFIX)
endif()

option(${OPTION_PREFIX}_PYTHON_INPLACE_BUILD "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)
option(${OPTION_PREFIX}_PYTHON_PERFORM_INSTALL "Whether or not to automatically `pip install` any built python library. WARNING: This may overwrite any existing installation of the same name." OFF)
option(${OPTION_PREFIX}_PYTHON_BUILD_STUBS "Whether or not to generated .pyi stub files for C++ Python modules. Disable to avoid requiring loading the NVIDIA GPU Driver during build" ON)

set(Python3_FIND_VIRTUALENV "FIRST")
set(Python3_FIND_STRATEGY "LOCATION")

message(VERBOSE "Python3_EXECUTABLE (before find_package): ${Python3_EXECUTABLE}")
message(VERBOSE "Python3_ROOT_DIR (before find_package): ${Python3_ROOT_DIR}")
message(VERBOSE "FIND_PYTHON_STRATEGY (before find_package): ${FIND_PYTHON_STRATEGY}")

find_package(Python3 REQUIRED COMPONENTS Development Interpreter)

message(VERBOSE "Python3_FOUND: " ${Python3_FOUND})
message(VERBOSE "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(VERBOSE "Python3_INTERPRETER_ID: " ${Python3_INTERPRETER_ID})
message(VERBOSE "Python3_STDLIB: " ${Python3_STDLIB})
message(VERBOSE "Python3_STDARCH: " ${Python3_STDARCH})
message(VERBOSE "Python3_SITELIB: " ${Python3_SITELIB})
message(VERBOSE "Python3_SITEARCH: " ${Python3_SITEARCH})
message(VERBOSE "Python3_SOABI: " ${Python3_SOABI})
message(VERBOSE "Python3_INCLUDE_DIRS: " ${Python3_INCLUDE_DIRS})
message(VERBOSE "Python3_LIBRARIES: " ${Python3_LIBRARIES})
message(VERBOSE "Python3_LIBRARY_DIRS: " ${Python3_LIBRARY_DIRS})
message(VERBOSE "Python3_VERSION: " ${Python3_VERSION})
message(VERBOSE "Python3_NumPy_FOUND: " ${Python3_NumPy_FOUND})
message(VERBOSE "Python3_NumPy_INCLUDE_DIRS: " ${Python3_NumPy_INCLUDE_DIRS})
message(VERBOSE "Python3_NumPy_VERSION: " ${Python3_NumPy_VERSION})

# After finding python, now find pybind11

# pybind11
# =========
set(PYBIND11_VERSION "2.8.1" CACHE STRING "Version of Pybind11 to use")
include(deps/Configure_pybind11)

if (NOT EXISTS ${Python3_SITELIB}/skbuild)
    # In case this is messed up by `/usr/local/python/site-packages` vs `/usr/python/site-packages`, check pip itself.
    execute_process(
        COMMAND bash "-c" "${Python3_EXECUTABLE} -m pip show scikit-build | sed -n -e 's/Location: //p'"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT EXISTS ${PYTHON_SITE_PACKAGES}/skbuild)
        message(SEND_ERROR "Scikit-build is not installed. CMake may not be able to find Cython. Install scikit-build with `pip install scikit-build`")
    else()
        list(APPEND CMAKE_MODULE_PATH "${PYTHON_SITE_PACKAGES}/skbuild/resources/cmake")
    endif()
else ()
    list(APPEND CMAKE_MODULE_PATH "${Python3_SITELIB}/skbuild/resources/cmake")
endif ()

set(CYTHON_FLAGS "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True" CACHE STRING "The directives for Cython compilation.")

# Now we can find pybind11
find_package(pybind11 REQUIRED)
find_package(Cython REQUIRED)


function(create_python_package PACKAGE_NAME)

  list(APPEND CMAKE_MESSAGE_CONTEXT "${PACKAGE_NAME}")
  set(CMAKE_MESSAGE_CONTEXT ${CMAKE_MESSAGE_CONTEXT} PARENT_SCOPE)

  if(PYTHON_ACTIVE_PACKAGE_NAME)
    message(FATAL_ERROR "An active wheel has already been created. Must call create_python_package/build_python_package in pairs")
  endif()

  message(STATUS "Creating python package '${PACKAGE_NAME}'")

  # Set the active wheel in the parent scipe
  set(PYTHON_ACTIVE_PACKAGE_NAME ${PACKAGE_NAME}-package)

  # Create a dummy source that holds all of the source files as resources
  add_custom_target(${PYTHON_ACTIVE_PACKAGE_NAME}-sources ALL)

  # Make it depend on the sources
  add_custom_target(${PYTHON_ACTIVE_PACKAGE_NAME}-modules ALL
    DEPENDS ${PYTHON_ACTIVE_PACKAGE_NAME}-sources
  )

  # Outputs target depends on all sources, generated files, and modules
  add_custom_target(${PYTHON_ACTIVE_PACKAGE_NAME}-outputs ALL
    DEPENDS ${PYTHON_ACTIVE_PACKAGE_NAME}-modules
  )

  # Now setup some simple globbing for common files to move to the build directory
  file(GLOB_RECURSE wheel_python_files
    LIST_DIRECTORIES FALSE
    CONFIGURE_DEPENDS
    "*.py"
    "py.typed"
    "pyproject.toml"
    "setup.cfg"
    "MANIFEST.in"
  )

  add_python_sources(${wheel_python_files})

  # Set the active wheel in the parent scope so it will appear in any subdirectories
  set(PYTHON_ACTIVE_PACKAGE_NAME ${PYTHON_ACTIVE_PACKAGE_NAME} PARENT_SCOPE)

endfunction()

function(add_target_resources)

  set(flags "")
  set(singleValues TARGET_NAME)
  set(multiValues "")

  include(CMakeParseArguments)
  cmake_parse_arguments(_ARGS
    "${flags}"
    "${singleValues}"
    "${multiValues}"
    ${ARGN}
  )

  # Get the current target resources
  get_target_property(target_resources ${_ARGS_TARGET_NAME} RESOURCE)

  set(args_absolute_paths)

  foreach(resource ${_ARGS_UNPARSED_ARGUMENTS})

    cmake_path(ABSOLUTE_PATH resource NORMALIZE OUTPUT_VARIABLE resource_absolute)

    list(APPEND args_absolute_paths "${resource_absolute}")

  endforeach()

  if(target_resources)
    # Append the list of supplied resources
    list(APPEND target_resources "${args_absolute_paths}")
  else()
    set(target_resources "${args_absolute_paths}")
  endif()

  set_target_properties(${_ARGS_TARGET_NAME} PROPERTIES RESOURCE "${target_resources}")

endfunction()

function(add_python_sources)

  if(NOT PYTHON_ACTIVE_PACKAGE_NAME)
    message(FATAL_ERROR "Must call create_python_wheel() before calling add_python_sources")
  endif()

  # Append any arguments to the python_sources_target
  add_target_resources(TARGET_NAME ${PYTHON_ACTIVE_PACKAGE_NAME}-sources ${ARGN})

endfunction()

function(copy_target_resources TARGET_NAME COPY_DIRECTORY)

  # See if there are any resources associated with this target
  get_target_property(target_resources ${TARGET_NAME} RESOURCE)

  if(target_resources)

    # Get the build and src locations
    get_target_property(target_source_dir ${TARGET_NAME} SOURCE_DIR)
    get_target_property(target_binary_dir ${TARGET_NAME} BINARY_DIR)

    set(resource_outputs "")

    # Create the copy command for each resource
    foreach(resource ${target_resources})

      # Get the absolute path of the resource in case its relative.
      cmake_path(ABSOLUTE_PATH resource NORMALIZE)

      cmake_path(IS_PREFIX target_source_dir "${resource}" NORMALIZE is_source_relative)
      cmake_path(IS_PREFIX target_binary_dir "${resource}" NORMALIZE is_binary_relative)

      # Get the relative path to the source or binary directories
      if(is_binary_relative)
        # This must come first because build is relative to source
        cmake_path(RELATIVE_PATH resource BASE_DIRECTORY "${target_binary_dir}" OUTPUT_VARIABLE resource_relative)
      elseif(is_source_relative)
        cmake_path(RELATIVE_PATH resource BASE_DIRECTORY "${target_source_dir}" OUTPUT_VARIABLE resource_relative)
      else()
        message(SEND_ERROR "Target resource is not relative to the source or binary directory. Target: ${TARGET_NAME}, Resource: ${resource}")
      endif()

      # Get the final copied location
      cmake_path(APPEND COPY_DIRECTORY "${resource_relative}" OUTPUT_VARIABLE resource_output)

      message(VERBOSE "Copying ${resource} to ${resource_output}")

      # Pretty up the output message
      set(top_level_source_dir ${${CMAKE_PROJECT_NAME}_SOURCE_DIR})
      cmake_path(RELATIVE_PATH resource BASE_DIRECTORY "${top_level_source_dir}" OUTPUT_VARIABLE resource_source_relative)
      cmake_path(RELATIVE_PATH resource_output BASE_DIRECTORY "${top_level_source_dir}" OUTPUT_VARIABLE resource_output_source_relative)

      add_custom_command(
        OUTPUT  ${resource_output}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${resource} ${resource_output}
        DEPENDS ${resource}
        COMMENT "Copying \${SOURCE_DIR}/${resource_source_relative} to \${SOURCE_DIR}/${resource_output_source_relative}"
      )

      list(APPEND resource_outputs ${resource_output})
    endforeach()

    # Final target to depend on the copied files
    add_custom_target(${TARGET_NAME}-copy-resources ALL
      DEPENDS ${resource_outputs}
    )
  endif()

endfunction()

function(inplace_build_copy TARGET_NAME INPLACE_DIR)
  message(VERBOSE "Inplace build: (${TARGET_NAME}) ${INPLACE_DIR}")

  add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${INPLACE_DIR}
    COMMENT "Moving target ${TARGET_NAME} to ${INPLACE_DIR} for inplace build"
  )

  copy_target_resources(${TARGET_NAME} ${INPLACE_DIR})

endfunction()

function(build_python_package PACKAGE_NAME)

  if(NOT PYTHON_ACTIVE_PACKAGE_NAME)
    message(FATAL_ERROR "Must call create_python_package() before calling add_python_sources")
  endif()

  if(NOT "${PACKAGE_NAME}-package" STREQUAL "${PYTHON_ACTIVE_PACKAGE_NAME}")
    message(FATAL_ERROR "Mismatched package name supplied to create_python_package/build_python_package")
  endif()

  set(flags BUILD_WHEEL INSTALL_WHEEL IS_INPLACE)
  set(singleValues "")
  set(multiValues PYTHON_DEPENDENCIES)

  include(CMakeParseArguments)
  cmake_parse_arguments(_ARGS
    "${flags}"
    "${singleValues}"
    "${multiValues}"
    ${ARGN}
  )

  message(STATUS "Finalizing python package '${PACKAGE_NAME}'")

  get_target_property(sources_source_dir ${PYTHON_ACTIVE_PACKAGE_NAME}-sources SOURCE_DIR)
  get_target_property(sources_binary_dir ${PYTHON_ACTIVE_PACKAGE_NAME}-sources BINARY_DIR)

  # First copy the source files
  copy_target_resources(${PYTHON_ACTIVE_PACKAGE_NAME}-sources ${sources_binary_dir})

  set(module_dependencies ${PYTHON_ACTIVE_PACKAGE_NAME}-sources-copy-resources)

  if(_ARGS_PYTHON_DEPENDENCIES)
    list(APPEND module_dependencies ${_ARGS_PYTHON_DEPENDENCIES})
  endif()

  # Now ensure that the targets only get built after the files have been copied
  add_dependencies(${PYTHON_ACTIVE_PACKAGE_NAME}-modules ${module_dependencies})

  # Next step is to build the wheel file
  if(_ARGS_BUILD_WHEEL)
    set(wheel_stamp ${sources_binary_dir}/${PYTHON_ACTIVE_PACKAGE_NAME}-wheel.stamp)

    # The command to actually generate the wheel
    add_custom_command(
      OUTPUT ${wheel_stamp}
      COMMAND python setup.py bdist_wheel
      COMMAND ${CMAKE_COMMAND} -E touch ${wheel_stamp}
      WORKING_DIRECTORY ${sources_binary_dir}
      # Depend on any of the output python files
      DEPENDS ${PYTHON_ACTIVE_PACKAGE_NAME}-outputs
      COMMENT "Building ${PACKAGE_NAME} wheel"
    )

    # Create a dummy target to ensure the above custom command is always run
    add_custom_target(${PYTHON_ACTIVE_PACKAGE_NAME}-wheel ALL
      DEPENDS ${install_stamp}
    )

    message(STATUS "Creating python wheel for library '${PACKAGE_NAME}'")
  endif()

  # Now build up the pip arguments to either install the package or print a message with the install command
  set(_pip_command)

  list(APPEND _pip_command  "${Python3_EXECUTABLE}" "-m" "pip" "install")

  # detect virtualenv and set Pip args accordingly
  if(NOT DEFINED ENV{VIRTUAL_ENV} AND NOT DEFINED ENV{CONDA_PREFIX})
    list(APPEND _pip_command  "--user")
  endif()

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    list(APPEND _pip_command "-e")
  endif()

  # Change which setup we use if we are using inplace
  if(_ARGS_IS_INPLACE)
    list(APPEND _pip_command "${sources_source_dir}")
  else()
    list(APPEND _pip_command "${sources_binary_dir}")
  endif()

  if(_ARGS_INSTALL_WHEEL)
    message(STATUS "Automatically installing Python package '${PACKAGE_NAME}' into current python environment. This may overwrite any existing library with the same name")

    # Now actually install the package
    set(install_stamp ${sources_binary_dir}/${PYTHON_ACTIVE_PACKAGE_NAME}-install.stamp)
    set(install_stamp_depfile ${install_stamp}.d)

    add_custom_command(
      OUTPUT ${install_stamp}
      COMMAND ${_pip_command}
      COMMAND ${CMAKE_COMMAND} -E touch ${install_stamp}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/pip_gen_depfile.py --pkg_name ${PACKAGE_NAME} --input_file ${install_stamp} --output_file ${install_stamp_depfile}
      DEPENDS ${PYTHON_ACTIVE_PACKAGE_NAME}-outputs
      DEPFILE ${install_stamp_depfile}
      COMMENT "Installing ${PACKAGE_NAME} python package"
    )

    add_custom_target(${PYTHON_ACTIVE_PACKAGE_NAME}-install ALL
      DEPENDS ${install_stamp}
    )
  else()
    list(JOIN _pip_command " " _pip_command_str)
    message(STATUS "Python package '${PACKAGE_NAME}' has been built but has not been installed. Use `${_pip_command_str}` to install the library manually")
  endif()

  # Finally, unset the active package
  unset(PYTHON_ACTIVE_PACKAGE_NAME PARENT_SCOPE)

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)
  set(CMAKE_MESSAGE_CONTEXT ${CMAKE_MESSAGE_CONTEXT} PARENT_SCOPE)

endfunction()


#[=======================================================================[
@brief : given a module name, and potentially a root path, resolves the
fully qualified python module path. If MODULE_ROOT is not provided, it
will default to ${CMAKE_CURRENT_SOURCE_DIR} -- the context of
the caller.

ex. resolve_python_module_name(my_module MODULE_ROOT morpheus/_lib)
results --
  MODULE_TARGET_NAME:   morpheus._lib.my_module
  OUTPUT_MODULE_NAME:   my_module
  OUTPUT_RELATIVE_PATH: morpheus/_lib

resolve_python_module_name <MODULE_NAME>
                           [MODULE_ROOT]
                           [OUTPUT_TARGET_NAME]
                           [OUTPUT_MODULE_NAME]
                           [OUTPUT_RELATIVE_PATH]
#]=======================================================================]

function(resolve_python_module_name MODULE_NAME)
  set(prefix _ARGS) # Prefix parsed args
  set(flags "")
  set(singleValues
      MODULE_ROOT
      OUTPUT_TARGET_NAME
      OUTPUT_MODULE_NAME
      OUTPUT_RELATIVE_PATH)
  set(multiValues "")

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${flags}"
      "${singleValues}"
      "${multiValues}"
      ${ARGN})

  set(py_module_name ${MODULE_NAME})
  set(py_module_namespace "")
  set(py_module_path "")

  if(_ARGS_MODULE_ROOT)
    file(RELATIVE_PATH py_module_path ${_ARGS_MODULE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})

    if(NOT ${py_module_path} STREQUAL "")
      # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
      # Always add a trailing / to ensure we end with a .
      string(REPLACE "/" "." py_module_namespace "${py_module_path}/")
    endif()
  endif()

  if (_ARGS_OUTPUT_TARGET_NAME)
    set(${_ARGS_OUTPUT_TARGET_NAME} "${py_module_namespace}${py_module_name}" PARENT_SCOPE)
  endif()
  if (_ARGS_OUTPUT_MODULE_NAME)
    set(${_ARGS_OUTPUT_MODULE_NAME} "${py_module_name}" PARENT_SCOPE)
  endif()
  if (_ARGS_OUTPUT_RELATIVE_PATH)
    set(${_ARGS_OUTPUT_RELATIVE_PATH} "${py_module_path}" PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[
@brief : TODO
ex. add_python_module
results --

add_python_module

#]=======================================================================]
macro(_create_python_library MODULE_NAME)

  list(APPEND CMAKE_MESSAGE_CONTEXT "${MODULE_NAME}")

  if(NOT PYTHON_ACTIVE_PACKAGE_NAME)
    message(FATAL_ERROR "Must call create_python_wheel() before calling add_python_sources")
  endif()

  set(prefix _ARGS)
  set(flags IS_PYBIND11 IS_CYTHON IS_MODULE COPY_INPLACE BUILD_STUBS)
  set(singleValues INSTALL_DEST OUTPUT_TARGET MODULE_ROOT PYX_FILE)
  set(multiValues INCLUDE_DIRS LINK_TARGETS SOURCE_FILES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${flags}"
      "${singleValues}"
      "${multiValues}"
      ${ARGN})

  if(NOT _ARGS_MODULE_ROOT)
    get_target_property(_ARGS_MODULE_ROOT ${PYTHON_ACTIVE_PACKAGE_NAME}-modules SOURCE_DIR)
  endif()

  # Normalize the module root
  cmake_path(SET _ARGS_MODULE_ROOT "${_ARGS_MODULE_ROOT}")

  resolve_python_module_name(${MODULE_NAME}
    MODULE_ROOT ${_ARGS_MODULE_ROOT}
    OUTPUT_TARGET_NAME TARGET_NAME
    OUTPUT_MODULE_NAME MODULE_NAME
    OUTPUT_RELATIVE_PATH SOURCE_RELATIVE_PATH
  )

  set(lib_type SHARED)

  if(_ARGS_IS_MODULE)
    set(lib_type MODULE)
  endif()

  # Create the module target
  if(_ARGS_IS_PYBIND11)
    message(VERBOSE "Adding Pybind11 Module: ${TARGET_NAME}")
    pybind11_add_module(${TARGET_NAME} ${lib_type} ${_ARGS_SOURCE_FILES})
  elseif(_ARGS_IS_CYTHON)
    message(VERBOSE "Adding Cython Module: ${TARGET_NAME}")
    add_cython_target(${MODULE_NAME} "${_ARGS_PYX_FILE}" CXX PY3)
    add_library(${TARGET_NAME} ${lib_type} ${${MODULE_NAME}} ${_ARGS_SOURCE_FILES})

    # Need to set -fvisibility=hidden for cython according to https://pybind11.readthedocs.io/en/stable/faq.html
    # set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  else()
    message(FATAL_ERROR "Must specify either IS_PYBIND11 or IS_CYTHON")
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")

  set(_link_libs "")
  if(_ARGS_LINK_TARGETS)
    foreach(target IN LISTS _ARGS_LINK_TARGETS)
      list(APPEND _link_libs ${target})
    endforeach()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${_link_libs}
  )

  # Tell CMake to use relative paths in the build directory. This is necessary for relocatable packages
  set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN")

  if(_ARGS_INCLUDE_DIRS)
    target_include_directories(${TARGET_NAME}
      PRIVATE
        "${_ARGS_INCLUDE_DIRS}"
    )
  endif()

  # Cython targets need the current dir for generated files
  if(_ARGS_IS_CYTHON)
    target_include_directories(${TARGET_NAME}
      PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}"
    )
  endif()

  # Set all_python_targets to depend on this module. This ensures that all python targets have been built before any
  # post build actions are taken. This is often necessary to allow post build actions that load the python modules to
  # succeed
  add_dependencies(${PYTHON_ACTIVE_PACKAGE_NAME}-modules ${TARGET_NAME})

  if(_ARGS_BUILD_STUBS)
    # Get the relative path from the project source to the module root
    cmake_path(RELATIVE_PATH _ARGS_MODULE_ROOT BASE_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE module_root_relative)

    cmake_path(APPEND PROJECT_BINARY_DIR ${module_root_relative} OUTPUT_VARIABLE module_root_binary_dir)
    cmake_path(APPEND module_root_binary_dir ${SOURCE_RELATIVE_PATH} ${MODULE_NAME} "__init__.pyi" OUTPUT_VARIABLE module_binary_stub_file)

    # Before installing, create the custom command to generate the stubs
    add_custom_command(
      OUTPUT  ${module_binary_stub_file}
      COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen ${TARGET_NAME} --no-setup-py --log-level WARN -o ./ --root-module-suffix \"\"
      DEPENDS ${PYTHON_ACTIVE_PACKAGE_NAME}-modules $<TARGET_OBJECTS:${TARGET_NAME}>
      COMMENT "Building stub for python module ${TARGET_NAME}..."
      WORKING_DIRECTORY ${module_root_binary_dir}
    )

    # Add a custom target to ensure the stub generation runs
    add_custom_target(${TARGET_NAME}-stubs ALL
      DEPENDS ${module_binary_stub_file}
    )

    # Make the outputs depend on the stub
    add_dependencies(${PYTHON_ACTIVE_PACKAGE_NAME}-outputs ${TARGET_NAME}-stubs)

    # Save the output as a target property
    add_target_resources(TARGET_NAME ${TARGET_NAME} "${module_binary_stub_file}")
  endif()

  if(_ARGS_INSTALL_DEST)
    message(VERBOSE "Install dest: (${TARGET_NAME}) ${_ARGS_INSTALL_DEST}")
    install(
      TARGETS
        ${TARGET_NAME}
      EXPORT
        ${PROJECT_NAME}-exports
      LIBRARY
        DESTINATION
          "${_ARGS_INSTALL_DEST}"
        COMPONENT Wheel
      RESOURCE
        DESTINATION
          "${_ARGS_INSTALL_DEST}/${MODULE_NAME}"
        COMPONENT Wheel
    )
  endif()

  # Set the output target
  if(_ARGS_OUTPUT_TARGET)
    set(${_ARGS_OUTPUT_TARGET} "${TARGET_NAME}" PARENT_SCOPE)
  endif()

  if(_ARGS_COPY_INPLACE)
    # Copy the target inplace
    inplace_build_copy(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  list(POP_BACK CMAKE_MESSAGE_CONTEXT)

endmacro()

#[=======================================================================[
@brief : TODO
ex. add_cython_library
results --

add_cython_library

#]=======================================================================]
function(add_cython_library MODULE_NAME)

  _create_python_library(${MODULE_NAME} IS_CYTHON ${ARGN})

endfunction()

#[=======================================================================[
@brief : TODO
ex. add_pybind11_module
results --

add_pybind11_module

#]=======================================================================]
function(add_pybind11_module MODULE_NAME)

  # Add IS_MODULE to make a MODULE instead of a SHARED
  _create_python_library(${MODULE_NAME} IS_PYBIND11 IS_MODULE ${ARGN})

endfunction()

#[=======================================================================[
@brief : TODO
ex. add_pybind11_library
results --

add_pybind11_library

#]=======================================================================]
function(add_pybind11_library MODULE_NAME)

  _create_python_library(${MODULE_NAME} IS_PYBIND11 ${ARGN})

endfunction()
