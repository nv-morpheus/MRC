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
def test_dynamic_module_registration():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{DYN_LIB_DIR}")
    plugin_module.load()

    module_namespace = "srf_unittest_cpp_dynamic"
    module_name = "DynamicSourceModule"

    registry = srf.ModuleRegistry

    assert registry.contains_namespace(module_namespace)
    assert registry.contains(module_name, module_namespace)

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(input):
            global packet_count
            packet_count += 1

        def on_error():
            pass

        def on_complete():
            pass

        config = {"source_count": 42}

        dynamic_source_mod = builder.load_module("DynamicSourceModule",
                                                 "srf_unittest_cpp_dynamic",
                                                 "DynamicModuleSourceTest_mod1",
                                                 config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(dynamic_source_mod.output_port("source"), sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("DynamicSourceModule_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42
    assert plugin_module.unload()


@pytest.mark.skipif(not FOUND_DYN_LIB, reason="Missing: libdynamic_test_module.so")
def test_dynamic_module_plugin_registration():
    plugin_module = srf.PluginModule.create_or_acquire("libdynamic_test_module.so")
    plugin_module.set_library_directory(f"{DYN_LIB_DIR}")
    plugin_module.load()

    module_namespace = "srf_unittest_cpp_dynamic"

    module_name = "DynamicSourceModule"

    registry = srf.ModuleRegistry

    assert registry.contains_namespace(module_namespace)
    assert registry.contains(module_name, module_namespace)

    registered_modules = registry.registered_modules()

    assert "srf_unittest_cpp_dynamic" in registered_modules

    ns_1 = registered_modules["srf_unittest_cpp_dynamic"]
    assert len(ns_1) == 1
    assert ns_1[0] == "DynamicSourceModule"

    assert "srf_unittest_cpp_dynamic_2" in registered_modules

    ns_2 = registered_modules["srf_unittest_cpp_dynamic_2"]
    assert len(ns_2) == 1
    assert ns_2[0] == "DynamicSourceModule"

    assert "srf_unittest_cpp_dynamic_3" in registered_modules

    ns_3 = registered_modules["srf_unittest_cpp_dynamic_3"]
    assert len(ns_3) == 1
    assert ns_3[0] == "DynamicSourceModule"

    actual_modules = plugin_module.list_modules()
    assert len(actual_modules) == 3

    expected_modules = [
        "srf_unittest_cpp_dynamic::DynamicSourceModule",
        "srf_unittest_cpp_dynamic_2::DynamicSourceModule",
        "srf_unittest_cpp_dynamic_3::DynamicSourceModule"
    ]

    assert actual_modules == expected_modules

    plugin_module.unload()

    registered_modules = registry.registered_modules()

    assert "srf_unittest_cpp_dynamic" not in registered_modules
    assert "srf_unittest_cpp_dynamic_2" not in registered_modules
    assert "srf_unittest_cpp_dynamic_3" not in registered_modules


@pytest.mark.skipif(not FOUND_DYN_LIB, reason="Failed to find libdynamic_test_module.so")
def test_dynamic_module_bad_version_test():

    BAD_VERSION = [13, 14, 15]
    module_name = "DynamicSourceModule_BAD"
    module_namespace = "srf_unittest_cpp_dynamic_BAD"

    def module_initializer(builder: srf.Builder):
        config = {"source_count": 42}

        builder.load_module("DynamicSourceModule_BAD",
                            "srf_unittest_cpp_dynamic_BAD",
                            "DynamicSourceModule_BAD_Test",
                            config)

    registry = srf.ModuleRegistry

    with pytest.raises(Exception):
        registry.register_module(module_name, module_namespace, BAD_VERSION, module_initializer)

    assert registry.contains_namespace(module_namespace) is not True
    assert registry.contains(module_namespace, module_namespace) is not True


if (__name__ in ("__main__", )):
    test_plugin_module_create_or_acquire()
    test_dynamic_module_plugin_interface()
    test_dynamic_module_plugin_registration()
    test_dynamic_module_bad_version_test()
    test_dynamic_module_registration()
