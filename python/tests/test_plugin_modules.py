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

import glob
import pathlib

import pytest
import srf

whereami = pathlib.Path(__file__).parent.resolve()

FOUND_DYN_LIB = False
DYN_LIB_DIR = None

dyn_lib_candidates = list(glob.iglob(f"{whereami}/../../**/libdynamic_test_module.so", recursive=True))
if (len(dyn_lib_candidates) >= 1):
    FOUND_DYN_LIB = True
    DYN_LIB_DIR = pathlib.Path(dyn_lib_candidates[0]).parent.resolve()


def test_plugin_module_create_or_acquire():
    mod = srf.PluginModule.create_or_acquire("doesnt_exist.so")

    assert mod is not None


@pytest.mark.skipif(not FOUND_DYN_LIB, reason="Missing: libdynamic_test_module.so")
def test_dynamic_module_plugin_interface():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{DYN_LIB_DIR}")
    plugin_module.load()
    plugin_module.reload()


@pytest.mark.skipif(not FOUND_DYN_LIB, reason="Missing: libdynamic_test_module.so")
def test_dynamic_module_plugin_registration():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{DYN_LIB_DIR}")
    plugin_module.load()

    # module_namespace = "srf_unittest_cpp_dynamic"

    # module_name = "DynamicSourceModule"

    # TODO(bhargav) add tests to ensure that dynamic modules are loaded correctly once registry bindings are in


@pytest.mark.skipif(not FOUND_DYN_LIB, reason="Failed to find libdynamic_test_module.so")
def test_dynamic_module_bad_version_test():
    pass


if (__name__ in ("__main__",)):
    test_plugin_module_create_or_acquire()
    test_dynamic_module_plugin_interface()
    test_dynamic_module_plugin_registration()
