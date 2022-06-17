# SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

list(APPEND CMAKE_MESSAGE_CONTEXT "python")

option(SRF_PYTHON_INPLACE_BUILD "Whether or not to copy built python modules back to the source tree for debug purposes." OFF)

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

# Now we can find pybind11
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Cython REQUIRED)

set(CYTHON_FLAGS "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True" CACHE STRING "The directives for Cython compilation.")

include(python_module_tools)

create_python_package(srf)

# First, lets find all python source files
# set(SRF_PYTHON_FILES "")

# # Get glob all of the python files
# file(GLOB_RECURSE SRF_PYTHON_FILES
#     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
#     CONFIGURE_DEPENDS
#     "*.py")

# # Manually install the below files. install(DIRECTORY) doesnt work well and
# # makes it impossible to get these files and srf/*.py in one command.
# list(PREPEND SRF_PYTHON_FILES
#     "pyproject.toml"
#     "setup.cfg"
#     "pytest.ini"
#     "tests/string_reader_input.txt"
#     # "setup.py"
#     # "versioneer.py"
# )

add_python_sources("pytest.ini" "tests/string_reader_input.txt")

# set(OUTPUT_PYTHON_FILES "")

# # For each file, copy if its different
# foreach(py_file ${SRF_PYTHON_FILES})

#     message(STATUS "Copying ${py_file} to ${CMAKE_CURRENT_BINARY_DIR}/${py_file}")

#     # Use configure file which will rerun if the input file changes
#     configure_file(
#         ${CMAKE_CURRENT_SOURCE_DIR}/${py_file}
#         ${CMAKE_CURRENT_BINARY_DIR}/${py_file}
#         COPYONLY
#     )

#     # add_custom_command(
#     #     OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${py_file}
#     #     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${py_file} ${CMAKE_CURRENT_BINARY_DIR}
#     #     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${py_file}
#     #     COMMAND "Copying ${py_file} to ${CMAKE_CURRENT_BINARY_DIR}/${py_file}"
#     # )

#     list(APPEND OUTPUT_PYTHON_FILES ${CMAKE_CURRENT_BINARY_DIR}/${py_file})

# endforeach()

# add_custom_target(srf_python_libraries
#     COMMENT "Running srf_python_libraries"
#     COMMAND ${CMAKE_COMMAND} -E echo "Running cmake srf_python_libraries")

# set(PYTHON_LIBRARY_STAMP ${CMAKE_CURRENT_BINARY_DIR}/library.stamp)

# add_custom_command(
#     COMMAND ${CMAKE_COMMAND} -E echo "Touching library stamp"
#     COMMAND ${CMAKE_COMMAND} -E touch ${PYTHON_LIBRARY_STAMP}
#     OUTPUT ${PYTHON_LIBRARY_STAMP}
# )

# add_custom_target(dest ALL DEPENDS ${PYTHON_LIBRARY_STAMP})

# Command to copy the python files to the build folder
# add_custom_target(srf_python_rebuild ALL
#     COMMENT "Building srf Wheel"
#     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_FILE_DIR:srf>/python
#     COMMAND python setup.py bdist_wheel
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
# )

# Save the root of the python for relative paths
set(SRF_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if (SRF_PYTHON_INPLACE_BUILD)
    message(STATUS "Performing inplace build for Python modules. Use `pip install -e .` from the python folder to install Srf.")
endif ()

# function(srf_get_python_module_name TARGET_NAME MODULE_NAME MODULE)
#     #get_filename_component(py_module_name ${CPP_FILENAME} NAME_WLE)

#     set(py_module_name ${MODULE})
#     # Get the relative path from the py root
#     file(RELATIVE_PATH py_rel_path ${SRF_PY_ROOT} ${CMAKE_CURRENT_SOURCE_DIR})
#     set(py_module_path ${py_rel_path})

#     # Convert the relative path to a namespace. i.e. `cuml/package/module` -> `cuml::package::module
#     string(REPLACE "/" "." py_module_namespace ${py_module_path})

#     set(${TARGET_NAME} "${py_module_namespace}.${py_module_name}" PARENT_SCOPE)
#     set(${MODULE_NAME} "${py_module_name}" PARENT_SCOPE)
#     #message("${py_module_namespace}.${py_module_name}")
# endfunction()

# function(srf_add_cython_libraries SOURCE_GLOB)

#     # Set the list of Cython files to build
#     file(GLOB cython_sources ${SOURCE_GLOB})

#     # Build all of the Cython targets
#     foreach (cpp_file ${cython_sources})
#         srf_get_python_module_name(TARGET_NAME MODULE_NAME ${cpp_file})

#         # Add the target and library
#         add_cython_target(${MODULE_NAME} "${cpp_file}" CXX PY3)
#         add_library(${TARGET_NAME} SHARED ${MODULE_NAME})

#         set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${cython_module_name}")

#         # Set props, include and link
#         set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
#         set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")
#         target_link_libraries(${TARGET_NAME} PUBLIC pysrf)
#         target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

#         if (SRF_PYTHON_INPLACE_BUILD)
#             # Get the source directory for the target
#             get_filename_component(cpp_source_dir ${cpp_file} DIRECTORY)

#             # If specified, copy files back to source for inplace build
#             add_custom_command(TARGET ${TARGET_NAME}
#                 POST_BUILD
#                 COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${cpp_source_dir}
#                 COMMENT "Moving python target ${TARGET_NAME} to ${cpp_source_dir} for inplace build"
#                 # BYPRODUCTS "${cpp_source_dir}/$<TARGET_FILE_NAME:${TARGET_NAME}>" # Uncomment if genex can be used in BYPRODUCTS. See https://gitlab.kitware.com/cmake/cmake/-/issues/21364
#             )
#         else()
#             # If we arent doing an inplace build, allow relocatable python build tree
#             set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
#         endif ()


#         # Install
#         # install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
#     endforeach (cpp_file ${cython_sources})

# endfunction()

# function(srf_add_pybind11_module)
#     set(moduleArg NAME)
#     set(sourceArgs SOURCE)
#     cmake_parse_arguments(ARGS "" "${moduleArg}" "${sourceArgs}" ${ARGN})

#     #message("=> ${ARGS_NAME}: ${ARGS_SOURCE}")

#     # Get the full module path
#     srf_get_python_module_name(TARGET_NAME MODULE_NAME ${ARGS_NAME})

#     # Add all globbed sources to the module sources
#     pybind11_add_module(${TARGET_NAME} MODULE "${ARGS_SOURCE}")

#     set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
#     set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${MODULE_NAME}")
#     target_link_libraries(${TARGET_NAME} PUBLIC pysrf)

#     if (SRF_PYTHON_INPLACE_BUILD)
#         foreach (source_file ${ARGS_SOURCE})

#             # Get the full path to the cpp file
#             file(REAL_PATH ${source_file} cpp_file)

#             # Get the source directory for the target
#             get_filename_component(cpp_source_dir ${cpp_file} DIRECTORY)

#             # If specified, copy files back to source for inplace build
#             add_custom_command(TARGET ${TARGET_NAME}
#                 POST_BUILD
#                 COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_NAME}> ${cpp_source_dir}
#                 COMMENT "Moving python target ${TARGET_NAME} to ${cpp_source_dir} for inplace build"
#                 # BYPRODUCTS "${cpp_source_dir}/$<TARGET_FILE_NAME:${TARGET_NAME}>" # Uncomment if genex can be used in BYPRODUCTS. See https://gitlab.kitware.com/cmake/cmake/-/issues/21364
#             )
#         endforeach ()
#     else()
#         # If we arent doing an inplace build, allow relocatable python build tree
#         set_target_properties(${TARGET_NAME} PROPERTIES BUILD_RPATH_USE_ORIGIN ON)
#     endif ()

#     # install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

#     add_dependencies(srf_python_libraries ${TARGET_NAME})

#     # add_custom_command(OUTPUT ${PYTHON_LIBRARY_STAMP} DEPENDS ${TARGET_NAME} APPEND)
# endfunction()

add_subdirectory(srf/_pysrf)
add_subdirectory(srf/core)
add_subdirectory(srf/benchmarking)

# # After all modules have been created
# set(PYTHON_WHEEL_STAMP ${CMAKE_CURRENT_BINARY_DIR}/wheel.stamp)

# # Create a custom command that depends on the srf_python_libraries and all .py files
# add_custom_command(
#     COMMAND python setup.py bdist_wheel
#     COMMAND ${CMAKE_COMMAND} -E touch ${PYTHON_WHEEL_STAMP}
#     OUTPUT ${PYTHON_WHEEL_STAMP}
#     WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#     # Depend on any of the output python files
#     DEPENDS ${OUTPUT_PYTHON_FILES} srf_python_libraries
# )

# # Create a dummy target to ensure the above custom command is always run
# add_custom_target(srf_python_rebuild ALL DEPENDS ${PYTHON_WHEEL_STAMP}
#     COMMENT "Running srf_python_rebuild"
#     COMMAND ${CMAKE_COMMAND} -E echo "Running cmake srf_python_rebuild")

###################################################################################################
# - tests -----------------------------------------------------------------------------------------

if (SRF_BUILD_TESTS)
    add_subdirectory(srf/tests)
endif ()

build_python_package(srf INSTALL_WHEEL $<$<BOOL:${SRF_PYTHON_INPLACE_BUILD}>:IS_INPLACE>)

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
