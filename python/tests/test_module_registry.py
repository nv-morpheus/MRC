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
import random

import pytest

import srf
import srf.tests.sample_modules

VERSION = [int(cmpt) for cmpt in srf.tests.sample_modules.__version__.split(".")]

packet_count = 0


def test_contains_namespace():
    registry = srf.ModuleRegistry

    assert registry.contains_namespace("xyz") is not True
    assert registry.contains_namespace("default")


def test_contains():
    registry = srf.ModuleRegistry

    assert registry.contains("SimpleModule", "srf_unittest")
    assert registry.contains("SourceModule", "srf_unittest")
    assert registry.contains("SinkModule", "srf_unittest")
    assert registry.contains("SimpleModule", "default") is not True


def test_is_version_compatible():
    registry = srf.ModuleRegistry

    release_version = [22, 11, 0]
    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True


def test_unregister_module():
    registry = srf.ModuleRegistry

    registry_namespace = "srf_unittest2"
    simple_mod_name = "SimpleModule"

    registry.unregister_module(simple_mod_name, registry_namespace)

    with pytest.raises(Exception):
        registry.unregister_module(simple_mod_name, registry_namespace, False)

    registry.unregister_module(simple_mod_name, registry_namespace, True)


def test_registered_modules():
    registry = srf.ModuleRegistry
    registered_mod_dict = registry.registered_modules()

    assert "default" in registered_mod_dict
    assert "srf_unittest" in registered_mod_dict
    assert len(registered_mod_dict) == 2


def module_init_fn(builder: srf.Builder):
    pass


def module_init_nested_fn(builder: srf.Builder):
    pass


# Purpose: Test basic dynamic module registration fails when given an incompatible version number
def test_module_registry_register_bad_version():
    registry = srf.ModuleRegistry

    # Bad version should result in a raised exception
    with pytest.raises(Exception):
        registry.register_module("a_module", "srf_unittests", [99, 99, 99], module_init_fn)


# Purpose: Test basic dynamic module registration and un-registration
def test_module_registry_register_good_version():
    registry = srf.ModuleRegistry

    registry.register_module("test_module_registry_register_good_version_module",
                             "srf_unittests",
                             VERSION,
                             module_init_fn)
    registry.unregister_module("test_module_registry_register_good_version_module", "srf_unittests")


# Purpose: Test basic dynamic module registration, and indirectly test correct shutdown/cleanup behavior
def test_module_registry_register_good_version_no_unregister():
    # Ensure that we don"t throw any errors or hang if we don"t explicitly unregister the python module
    registry = srf.ModuleRegistry

    registry.register_module("test_module_registry_register_good_version_no_unregister_module",
                             "srf_unittests",
                             VERSION,
                             module_init_fn)


def test_get_module_constructor():
    registry = srf.ModuleRegistry

    # Retrieve the module constructor
    fn_constructor = registry.get_module_constructor("SimpleModule", "srf_unittest")

    # Instantiate a version of the module
    config = {"config_key_1": True}
    module = fn_constructor("ModuleInitializationTest_mod", config)

    assert "config_key_1" in module.config()

    with pytest.raises(Exception):
        registry.get_module_constructor("SimpleModule", "default")


def test_module_intitialize():
    module_name = "test_py_source_from_cpp"
    config = {"source_count": 42}
    registry = srf.ModuleRegistry

    def module_initializer(builder: srf.Builder):
        local_config = builder.get_current_module_config()
        assert ("source_count" in local_config)
        assert (local_config["source_count"] == config["source_count"])

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", local_config)
        builder.register_module_output("source", source_mod.output_port("source"))

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        # Retrieve the module constructor
        fn_constructor = registry.get_module_constructor(module_name, "srf_unittest")
        # Instantiate a version of the module
        source_module = fn_constructor("ModuleSourceTest_mod1", config)

        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.init_module(source_module)
        builder.make_edge(source_module.output_port('source'), sink)

    # Register the module
    registry.register_module(module_name, "srf_unittest", VERSION, module_initializer)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42


# Purpose: Create a self-contained (no input/output ports), nested, dynamic module, and instantiate two copies in our
# init wrapper
def test_py_registered_nested_modules():
    global packet_count

    # Stand-alone module, no input or output ports
    # 1. We register a python module definition as being built by "init_registered"
    # 2. We then create a segment with a separate init function "init_caller" that loads our python module from
    #    the registry and initializes it, running
    def module_initializer(builder: srf.Builder):
        global packet_count

        local_config = builder.get_current_module_config()
        assert (isinstance(local_config, type({})))
        assert (len(local_config.keys()) == 0)

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        config = {"source_count": 42}

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)
        builder.make_edge(source_mod.output_port("source"), sink)

    def init_caller(builder: srf.Builder):
        global packet_count
        packet_count = 0
        builder.load_module("test_py_registered_nested_module", "srf_unittests", "my_loaded_module!", {})

    registry = srf.ModuleRegistry
    registry.register_module("test_py_registered_nested_module", "srf_unittests", VERSION, module_initializer)

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


# Purpose: Create a self-contained (no input/output ports), nested, dynamic module, and instantiate two copies in our
# init wrapper -- since both versions capture our global 'packet_count', we should see double the packets.
def test_py_registered_nested_copied_modules():
    global packet_count

    def module_initializer(builder: srf.Builder):
        local_config = builder.get_current_module_config()
        assert (isinstance(local_config, type({})))
        if ("test1" in local_config):
            assert ("test2" not in local_config)
            assert (local_config["test1"] == "module_1")
        else:
            assert ("test1" not in local_config)
            assert ("test2" in local_config)
            assert (local_config["test2"] == "module_2")

        global packet_count

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        config = {"source_count": 42}

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)
        builder.make_edge(source_mod.output_port("source"), sink)

    registry = srf.ModuleRegistry
    registry.register_module("test_py_registered_nested_copied_module", "srf_unittests", VERSION, module_initializer)

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0
        builder.load_module("test_py_registered_nested_copied_module",
                            "srf_unittests",
                            "my_loaded_module!", {"test1": "module_1"})
        builder.load_module("test_py_registered_nested_copied_module",
                            "srf_unittests",
                            "my_loaded_module_copy!", {"test2": "module_2"})

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    # We loaded two copies of the module, and each one captures the same packet_count, so we should see 2*42 packets
    assert packet_count == 84


# Test if we can create a [source_module] -> [sink] configuration, where the source module is a python source created
# via builder.make_source.
#
# Purpose: This is intended to check dynamic module creation, registration, and retrieval
def test_py_dynamic_module_source():
    global packet_count
    module_name = "test_py_dyn_source"

    def module_initializer(builder: srf.Builder):

        def gen_data():
            for x in range(42):
                yield random.choice([True, False])

        source1 = builder.make_source("dynamic_module_source", gen_data)
        builder.register_module_output("source", source1)

    registry = srf.ModuleRegistry
    registry.register_module(module_name, "srf_unittests", VERSION, module_initializer)

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        # Load our registered module
        source_mod = builder.load_module(module_name, "srf_unittests", "my_loaded_module!", {})
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source_mod.output_port("source"), sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42


# Test if we can create a [source_module] -> [sink] configuration, where the source module loads a c++ defined module
# from the registry and exposes its source as the source for our dynamic module.
#
# Purpose: This is intended to check dynamic module creation, registration, and retrieval, in conjunction with module
# nesting.
def test_py_dynamic_module_from_cpp_source():
    global packet_count
    module_name = "test_py_dyn_source_from_cpp"

    def module_initializer(builder: srf.Builder):
        config = {"source_count": 42}

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleSourceTest_mod1", config)
        builder.register_module_output("source", source_mod.output_port("source"))

    registry = srf.ModuleRegistry
    registry.register_module(module_name, "srf_unittests", VERSION, module_initializer)

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        # Load our registered module
        source_mod = builder.load_module(module_name, "srf_unittests", "my_loaded_module!", {})
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source_mod.output_port("source"), sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42


# Purpose: Test creation of a dynamic module that acts as a sink [source] -> [sink_module]
def test_py_dynamic_module_sink():
    global packet_count
    module_name = "test_py_dyn_sink"

    def module_initializer(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(data):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(data))

        def on_error():
            pass

        def on_complete():
            pass

        sink = builder.make_sink("sink", on_next, on_error, on_complete)
        builder.register_module_input("sink", sink)

    registry = srf.ModuleRegistry
    registry.register_module(module_name, "srf_unittests", VERSION, module_initializer)

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def gen_data():
            for x in range(42):
                yield random.choice([True, False])

        source = builder.make_source("source", gen_data)
        sink_mod = builder.load_module(module_name, "srf_unittests", "loaded_sink_module", {})

        builder.make_edge(source, sink_mod.input_port("sink"))

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42


# Purpose: Test creation of a dynamic module that acts as a sink [source] -> [sink_module], where sink module loads a
# c++ defined module from the registry and exposes its sink as the sink for the dynamic module.
def test_py_dynamic_module_from_cpp_sink():
    global packet_count
    module_name = "test_py_dyn_sink_from_cpp"

    def module_initializer(builder: srf.Builder):
        config = {"source_count": 42}

        sink_mod = builder.load_module("SinkModule", "srf_unittest", "ModuleSinkTest_Mod1", config)
        builder.register_module_input("sink", sink_mod.input_port("sink"))

    registry = srf.ModuleRegistry
    registry.register_module(module_name, "srf_unittests", VERSION, module_initializer)

    def gen_data():
        for x in range(42):
            yield random.choice([True, False])

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        source = builder.make_source("source", gen_data)
        sink_mod = builder.load_module(module_name, "srf_unittests", "loaded_sink_module", {})

        builder.make_edge(source, sink_mod.input_port("sink"))

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()


if (__name__ in ("__main__", )):
    test_module_intitialize()
    test_contains_namespace()
    test_contains()
    test_get_module_constructor()
    test_is_version_compatible()
    test_unregister_module()
    test_registered_modules()
    test_module_registry_register_bad_version()
    test_module_registry_register_good_version()
    test_module_registry_register_good_version_no_unregister()
    test_py_registered_nested_modules()
    test_py_registered_nested_copied_modules()
    test_py_dynamic_module_source()
    test_py_dynamic_module_from_cpp_source()
    test_py_dynamic_module_sink()
    test_py_dynamic_module_from_cpp_sink()
