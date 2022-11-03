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

import logging

import pytest

import srf
import srf.tests.sample_modules

VERSION = [int(cmpt) for cmpt in srf.tests.sample_modules.__version__.split(".")]


def test_contains_namespace():
    registry = srf.ModuleRegistry()

    assert registry.contains_namespace("xyz") is not True
    assert registry.contains_namespace("default")


def test_contains():
    registry = srf.ModuleRegistry()

    assert registry.contains("SimpleModule", "srf_unittest")
    assert registry.contains("SourceModule", "srf_unittest")
    assert registry.contains("SinkModule", "srf_unittest")
    assert registry.contains("SimpleModule", "default") is not True


def test_find_module():

    config = {"config_key_1": True}

    registry = srf.ModuleRegistry()

    simple_mod = registry.find_module("SimpleModule", "srf_unittest", "ModuleInitializationTest_mod", config)

    assert "config_key_1" in simple_mod.config()

    with pytest.raises(Exception):
        simple_mod = registry.find_module("SimpleModule", "default", "ModuleInitializationTest_mod", config)


def test_is_version_compatible():
    registry = srf.ModuleRegistry()

    release_version = [22, 11, 0]
    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True


def test_unregister_module():
    registry = srf.ModuleRegistry()

    registry_namespace = "srf_unittest2"
    simple_mod_name = "SimpleModule"

    registry.unregister_module(simple_mod_name, registry_namespace)

    with pytest.raises(Exception):
        registry.unregister_module(simple_mod_name, registry_namespace, False)

    registry.unregister_module(simple_mod_name, registry_namespace, True)


def test_registered_modules():
    registry = srf.ModuleRegistry()
    registered_mod_dict = registry.registered_modules()

    assert "default" in registered_mod_dict
    assert "srf_unittest" in registered_mod_dict
    assert len(registered_mod_dict) == 2


def module_init_fn(builder: srf.Builder, module: srf.SegmentModule):
    pass


def module_init_nested_fn(builder: srf.Builder, module: srf.SegmentModule):
    pass


def test_module_registry_register_bad_version():
    registry = srf.ModuleRegistry()

    # Bad version should result in a raised exception
    with pytest.raises(Exception):
        registry.register_module("a_module", "srf_unittests", [99, 99, 99], module_init_fn)


def test_module_registry_register_good_version():
    registry = srf.ModuleRegistry()

    registry.register_module("test_module_registry_register_good_version_module",
                             "srf_unittests",
                             VERSION,
                             module_init_fn)
    registry.unregister_module("test_module_registry_register_good_version_module", "srf_unittests")


def test_module_registry_register_good_version_no_unregister():
    # Ensure that we don"t throw any errors or hang if we don"t explicitly unregister the python module
    registry = srf.ModuleRegistry()

    registry.register_module("test_module_registry_register_good_version_no_unregister_module",
                             "srf_unittests",
                             VERSION,
                             module_init_fn)


def test_py_registered_nested_modules():
    # Stand-alone module, no input or output ports
    # 1. We register a python module definition as being built by "init_registered"
    # 2. We then create a segment with a separate init function "init_caller" that loads our python module from
    #    the registry and initializes it, running
    def init_registered(builder: srf.Builder):
        global packet_count

        def on_next(input):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(input))

        def on_error():
            pass

        def on_complete():
            pass

        config = {}
        config["source_count"] = 42

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)
        builder.make_edge(source_mod.output_port("source"), sink)

    def init_caller(builder: srf.Builder):
        global packet_count
        packet_count = 0
        builder.load_module("test_py_registered_nested_module", "srf_unittests", "my_loaded_module!", {})

    registry = srf.ModuleRegistry()
    registry.register_module("test_py_registered_nested_module", "srf_unittests", VERSION, init_registered)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_caller)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    # We loaded two copies of the module, and each one captures the same packet_count, so we should see 2*42 packets
    assert packet_count == 42


def test_py_registered_nested_copied_modules():
    # Stand-alone module, no input or output ports
    # 1. We register a python module definition as being built by "init_registered"
    # 2. We then create a segment with a separate init function "init_caller" that loads our python module from
    #    the registry and initializes it, running
    def init_registered(builder: srf.Builder):
        global packet_count

        def on_next(input):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(input))

        def on_error():
            pass

        def on_complete():
            pass

        config = {}
        config["source_count"] = 42

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)
        builder.make_edge(source_mod.output_port("source"), sink)

    def init_caller(builder: srf.Builder):
        global packet_count
        packet_count = 0
        builder.load_module("test_py_registered_nested_copied_module", "srf_unittests", "my_loaded_module!", {})
        builder.load_module("test_py_registered_nested_copied_module", "srf_unittests", "my_loaded_module_copy!", {})

    registry = srf.ModuleRegistry()
    registry.register_module("test_py_registered_nested_copied_module", "srf_unittests", VERSION, init_registered)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_caller)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    # We loaded two copies of the module, and each one captures the same packet_count, so we should see 2*42 packets
    assert packet_count == 84


if (__name__ in ("__main__", )):
    test_contains_namespace()
    test_contains()
    test_find_module()
    test_is_version_compatible()
    test_unregister_module()
    test_registered_modules()
    test_module_registry_register_bad_version()
    test_module_registry_register_good_version()
    test_module_registry_register_good_version_no_unregister()
    test_py_registered_nested_modules()
    test_py_registered_nested_copied_modules()
