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


# set(Python3_FIND_VIRTUALENV "FIRST")
# set(Python3_FIND_STRATEGY "LOCATION")

# message(VERBOSE "Python3_EXECUTABLE (before find_package): ${Python3_EXECUTABLE}")
# message(VERBOSE "Python3_ROOT_DIR (before find_package): ${Python3_ROOT_DIR}")
# message(VERBOSE "FIND_PYTHON_STRATEGY (before find_package): ${FIND_PYTHON_STRATEGY}")

# find_package(Python3 REQUIRED COMPONENTS Development Interpreter)

# message(VERBOSE "Python3_FOUND: " ${Python3_FOUND})
# message(VERBOSE "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
# message(VERBOSE "Python3_INTERPRETER_ID: " ${Python3_INTERPRETER_ID})
# message(VERBOSE "Python3_STDLIB: " ${Python3_STDLIB})
# message(VERBOSE "Python3_STDARCH: " ${Python3_STDARCH})
# message(VERBOSE "Python3_SITELIB: " ${Python3_SITELIB})
# message(VERBOSE "Python3_SITEARCH: " ${Python3_SITEARCH})
# message(VERBOSE "Python3_SOABI: " ${Python3_SOABI})
# message(VERBOSE "Python3_INCLUDE_DIRS: " ${Python3_INCLUDE_DIRS})
# message(VERBOSE "Python3_LIBRARIES: " ${Python3_LIBRARIES})
# message(VERBOSE "Python3_LIBRARY_DIRS: " ${Python3_LIBRARY_DIRS})
# message(VERBOSE "Python3_VERSION: " ${Python3_VERSION})
# message(VERBOSE "Python3_NumPy_FOUND: " ${Python3_NumPy_FOUND})
# message(VERBOSE "Python3_NumPy_INCLUDE_DIRS: " ${Python3_NumPy_INCLUDE_DIRS})
# message(VERBOSE "Python3_NumPy_VERSION: " ${Python3_NumPy_VERSION})

# # After finding python, now find pybind11

# # pybind11
# # =========
# set(PYBIND11_VERSION "2.8.1" CACHE STRING "Version of Pybind11 to use")
# include(deps/Configure_pybind11)

# if (NOT EXISTS ${Python3_SITELIB}/skbuild)
#     # In case this is messed up by `/usr/local/python/site-packages` vs `/usr/python/site-packages`, check pip itself.
#     execute_process(
#         COMMAND bash "-c" "${Python3_EXECUTABLE} -m pip show scikit-build | sed -n -e 's/Location: //p'"
#         OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#     )

#     if (NOT EXISTS ${PYTHON_SITE_PACKAGES}/skbuild)
#         message(SEND_ERROR "Scikit-build is not installed. CMake may not be able to find Cython. Install scikit-build with `pip install scikit-build`")
#     else()
#         list(APPEND CMAKE_MODULE_PATH "${PYTHON_SITE_PACKAGES}/skbuild/resources/cmake")
#     endif()
# else ()
#     list(APPEND CMAKE_MODULE_PATH "${Python3_SITELIB}/skbuild/resources/cmake")
# endif ()

# # Now we can find pybind11
# find_package(pybind11 REQUIRED)

# find_package(Cython REQUIRED)

# set(CYTHON_FLAGS "--directive binding=True,boundscheck=False,wraparound=False,embedsignature=True,always_allow_keywords=True" CACHE STRING "The directives for Cython compilation.")

find_package(CUDAToolkit REQUIRED)

include(python_module_tools)

# Create the srf python package
create_python_package(srf)

# Add a few additional files to be copied
add_python_sources("pytest.ini" "tests/string_reader_input.txt")

# Save the root of the python for relative paths
set(SRF_PY_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

if (SRF_PYTHON_INPLACE_BUILD)
    message(STATUS "Performing inplace build for Python modules. Use `pip install -e .` from the python folder to install Srf.")
endif ()

add_subdirectory(srf/_pysrf)
add_subdirectory(srf/core)
add_subdirectory(srf/benchmarking)

###################################################################################################
# - tests -----------------------------------------------------------------------------------------

if (SRF_BUILD_TESTS)
    add_subdirectory(srf/tests)
endif ()

# Complete the python package
set(extra_args "")

if(SRF_PYTHON_INPLACE_BUILD)
  message(STATUS "Doing inplace build")
  list(APPEND extra_args "IS_INPLACE")
endif()

if(SRF_PYTHON_PERFORM_INSTALL)
  message(STATUS "Doing install")
  list(APPEND extra_args "INSTALL_WHEEL")
endif()

message(STATUS "Final args: ${extra_args}")
build_python_package(srf ${extra_args})

list(POP_BACK CMAKE_MESSAGE_CONTEXT)
