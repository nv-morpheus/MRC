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
add_custom_target(all_python_targets ALL)

# This target is used to store all python files that need to be copied to the build directory as resources
add_custom_target(python_sources_target ALL)

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

  # Append any arguments to the python_sources_target
  add_target_resources(TARGET_NAME python_sources_target ${ARGN})

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

      set(daf ${PROJECT_IS_TOP_LEVEL})

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
  message(STATUS " Inplace build: (${TARGET_NAME}) ${INPLACE_DIR}")

  add_custom_command(
    TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${INPLACE_DIR}
    COMMENT "Moving target ${TARGET_NAME} to ${INPLACE_DIR} for inplace build"
  )

  copy_target_resources(${TARGET_NAME} ${INPLACE_DIR})

endfunction()


function(copy_python_sources)

  copy_target_resources(python_sources_target ${PROJECT_BINARY_DIR})

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
  set(prefix PYMOD) # Prefix parsed args
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

  if(PYMOD_MODULE_ROOT)
    file(RELATIVE_PATH py_module_path ${PYMOD_MODULE_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})

    if(NOT ${py_module_path} STREQUAL "")
      # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
      # Always add a trailing / to ensure we end with a .
      string(REPLACE "/" "." py_module_namespace "${py_module_path}/")
    endif()
  endif()

  if (PYMOD_OUTPUT_TARGET_NAME)
    set(${PYMOD_OUTPUT_TARGET_NAME} "${py_module_namespace}${py_module_name}" PARENT_SCOPE)
  endif()
  if (PYMOD_OUTPUT_MODULE_NAME)
    set(${PYMOD_OUTPUT_MODULE_NAME} "${py_module_name}" PARENT_SCOPE)
  endif()
  if (PYMOD_OUTPUT_RELATIVE_PATH)
    set(${PYMOD_OUTPUT_RELATIVE_PATH} "${py_module_path}" PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[
@brief : TODO
ex. add_python_module
results --

add_python_module

#]=======================================================================]
macro(_create_python_library MODULE_NAME)
  set(prefix PYMOD)
  set(flags IS_PYBIND11 IS_CYTHON IS_MODULE)
  set(singleValues INSTALL_DEST OUTPUT_TARGET MODULE_ROOT PYX_FILE)
  set(multiValues INCLUDE_DIRS LINK_TARGETS SOURCE_FILES)

  include(CMakeParseArguments)
  cmake_parse_arguments(${prefix}
      "${flags}"
      "${singleValues}"
      "${multiValues}"
      ${ARGN})

  if(PYMOD_MODULE_ROOT)
    cmake_path(SET PYMOD_MODULE_ROOT "${PYMOD_MODULE_ROOT}")
  endif()

  resolve_python_module_name(${MODULE_NAME}
    MODULE_ROOT ${PYMOD_MODULE_ROOT}
    OUTPUT_TARGET_NAME TARGET_NAME
    OUTPUT_MODULE_NAME MODULE_NAME
    OUTPUT_RELATIVE_PATH SOURCE_RELATIVE_PATH
  )

  # Ensure the custom target all_python_targets has been created
  if (NOT TARGET all_python_targets)
    message(FATAL_ERROR "You must call `add_custom_target(all_python_targets)` before the first call to add_python_module")
  endif()

  set(lib_type SHARED)

  if(PYMOD_IS_MODULE)
    set(lib_type MODULE)
  endif()

  # Create the module target
  if (PYMOD_IS_PYBIND11)
    message(STATUS "Adding Pybind11 Module: ${TARGET_NAME}")
    pybind11_add_module(${TARGET_NAME} ${lib_type} ${PYMOD_SOURCE_FILES})
  elseif(PYMOD_IS_CYTHON)
    message(STATUS "Adding Cython Module: ${TARGET_NAME}")
    add_cython_target(${MODULE_NAME} "${PYMOD_PYX_FILE}" CXX PY3)
    add_library(${TARGET_NAME} ${lib_type} ${${MODULE_NAME}} ${PYMOD_SOURCE_FILES})

    # Need to set -fvisibility=hidden for cython according to https://pybind11.readthedocs.io/en/stable/faq.html
    # set_target_properties(${TARGET_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
  else()
    message(FATAL_ERROR "Must specify either IS_PYBIND11 or IS_CYTHON")
  endif()

  set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
  set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")

  set(pymod_link_libs "")
  if (PYMOD_LINK_TARGETS)
    foreach(target IN LISTS PYMOD_LINK_TARGETS)
      list(APPEND pymod_link_libs ${target})
    endforeach()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${pymod_link_libs}
  )

  # Tell CMake to use relative paths in the build directory. This is necessary for relocatable packages
  set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib:\$ORIGIN")

  if (PYMOD_INCLUDE_DIRS)
    target_include_directories(${TARGET_NAME}
      PRIVATE
        "${PYMOD_INCLUDE_DIRS}"
    )
  endif()

  # Cython targets need the current dir for generated files
  if(PYMOD_IS_CYTHON)
    target_include_directories(${TARGET_NAME}
      PUBLIC
        "${CMAKE_CURRENT_BINARY_DIR}"
    )
  endif()

  # Set all_python_targets to depend on this module. This ensures that all python targets have been built before any
  # post build actions are taken. This is often necessary to allow post build actions that load the python modules to
  # succeed
  add_dependencies(all_python_targets ${TARGET_NAME})

  # Get the relative path from the project source to the module root
  cmake_path(RELATIVE_PATH PYMOD_MODULE_ROOT BASE_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE module_root_relative)

  cmake_path(APPEND PROJECT_BINARY_DIR ${module_root_relative} OUTPUT_VARIABLE module_root_binary_dir)
  cmake_path(APPEND module_root_binary_dir ${SOURCE_RELATIVE_PATH} ${MODULE_NAME} "__init__.pyi" OUTPUT_VARIABLE module_binary_stub_file)

  # Before installing, create the custom command to generate the stubs
  add_custom_command(
    OUTPUT  ${module_binary_stub_file}
    COMMAND ${Python3_EXECUTABLE} -m pybind11_stubgen ${TARGET_NAME} --no-setup-py --log-level WARN -o ./ --root-module-suffix \"\"
    DEPENDS ${TARGET_NAME} all_python_targets
    COMMENT "Building stub for python module ${TARGET_NAME}..."
    WORKING_DIRECTORY ${module_root_binary_dir}
  )

  # Add a custom target to ensure the stub generation runs
  add_custom_target(${TARGET_NAME}-stubs ALL
    DEPENDS ${module_binary_stub_file}
  )

  # Save the output as a target property
  add_target_resources(TARGET_NAME ${TARGET_NAME} "${module_binary_stub_file}")
  # set_target_properties(${TARGET_NAME} PROPERTIES RESOURCE "${module_binary_stub_file}")

  unset(module_binary_stub_file)

  if (PYMOD_INSTALL_DEST)
    message(STATUS " Install dest: (${TARGET_NAME}) ${PYMOD_INSTALL_DEST}")
    install(
      TARGETS
        ${TARGET_NAME}
      EXPORT
        ${PROJECT_NAME}-exports
      LIBRARY
        DESTINATION
          "${PYMOD_INSTALL_DEST}"
        COMPONENT Wheel
      RESOURCE
        DESTINATION
          "${PYMOD_INSTALL_DEST}/${MODULE_NAME}"
        COMPONENT Wheel
    )
  endif()

  # Set the output target
  if (PYMOD_OUTPUT_TARGET)
    set(${PYMOD_OUTPUT_TARGET} "${TARGET_NAME}" PARENT_SCOPE)
  endif()

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
