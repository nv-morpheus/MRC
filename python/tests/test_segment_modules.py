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
# Required to register sample modules with the ModuleRegistry
import srf.tests.sample_modules

packets_1 = 0
packets_2 = 0
packets_3 = 0


def test_py_end_to_end():

    def gen_data_1():
        yield True
        yield False
        yield True
        yield True

    def gen_data_2():
        yield True
        yield False
        yield False
        yield False
        yield True
        yield False

    #                                    Visualization of What's Going On
    #                                              SimpleModule
    #                                    __________________________________
    # source1 -> emitted boolean -> --- | input1 -- _internal1_ -- output1 | --- emitted string --- sink1
    #                                   |                                  |
    # source2 -> emitted boolean -> --- | input2 -- _internal2_ -- output2 | --- emitted string --- sink2
    #                                   |__________________________________|
    #
    #                                           ConfigurableModule
    #                                   ________________________________________________________________
    # source3 -> emitted boolean -> --- | configurable_input_a -- _internal1_ -- configurable_output_x | --- ... sink3
    #                                   |_______________________________________________________________
    #

    def init_wrapper(builder: srf.Builder):
        global packets_1, packets_2, packets_3
        packets_1, packets_2, packets_3 = 0, 0, 0

        def on_next_sink_1(input):
            global packets_1
            packets_1 += 1

        def on_next_sink_2(input):
            global packets_2
            packets_2 += 1

        def on_next_sink_3(input):
            global packets_3
            packets_3 += 1

        def on_error():
            pass

        def on_complete():
            pass

        simple_mod = builder.load_module("SimpleModule", "srf_unittest", "ModuleEndToEndTest_mod1", {})
        configurable_mod = builder.load_module("ConfigurableModule", "srf_unittest", "ModuleEndToEndTest_mod2", {})

        source1 = builder.make_source("src1", gen_data_1)
        builder.make_edge(source1, simple_mod.input_port("input1"))

        source2 = builder.make_source("src2", gen_data_2)
        builder.make_edge(source2, simple_mod.input_port("input2"))

        sink1 = builder.make_sink("sink1", on_next_sink_1, on_error, on_complete)
        builder.make_edge(simple_mod.output_port("output1"), sink1)

        sink2 = builder.make_sink("sink2", on_next_sink_2, on_error, on_complete)
        builder.make_edge(simple_mod.output_port("output2"), sink2)

        source3 = builder.make_source("src3", gen_data_1)
        builder.make_edge(source3, configurable_mod.input_port("configurable_input_a"))

        sink3 = builder.make_sink("sink3", on_next_sink_3, on_error, on_complete)
        builder.make_edge(configurable_mod.output_port("configurable_output_x"), sink3)

    pipe = srf.Pipeline()

    pipe.make_segment("EndToEnd_Segment", [], [], init_wrapper)

    options = srf.Options()

    executor = srf.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()

    assert (packets_1 == 4)
    assert (packets_2 == 6)
    assert (packets_3 == 4)


def test_py_constructor():

    config = {"config_key_1": True}

    registry = srf.ModuleRegistry

    # Retrieve the module constructor
    fn_constructor = registry.get_module_constructor("SimpleModule", "srf_unittest")

    # Instantiate a version of the module
    config = {"config_key_1": True}
    module = fn_constructor("ModuleInitializationTest_mod", config)

    assert "config_key_1" in module.config()

    with pytest.raises(Exception):
        registry.get_module_constructor("SimpleModule", "default")


def test_py_module_initialization():

    def gen_data():
        yield True
        yield False
        yield True
        yield True

    def init_wrapper(builder: srf.Builder):

        def on_next(input):
            pass

        def on_error():
            pass

        def on_complete():
            pass

        config = {"config_key_1": True}

        registry = srf.ModuleRegistry

        source = builder.make_source("source", gen_data)
        source2 = builder.make_source("source2", gen_data)
        fn_constructor = registry.get_module_constructor("SimpleModule", "srf_unittest")
        simple_mod = fn_constructor("ModuleInitializationTest_mod2", config)
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.init_module(simple_mod)

        assert len(simple_mod.input_ids()) == 2
        assert len(simple_mod.output_ids()) == 2
        assert len(simple_mod.input_ports()) == 2
        assert len(simple_mod.output_ports()) == 2

        assert ("input1" in simple_mod.input_ports())
        assert ("input2" in simple_mod.input_ports())
        assert ("output1" in simple_mod.output_ports())
        assert ("output2" in simple_mod.output_ports())

        with pytest.raises(Exception):
            simple_mod.input_port("DOES_NOT_EXIST")
        with pytest.raises(Exception):
            simple_mod.output_port("DOES_NOT_EXIST")
        with pytest.raises(Exception):
            simple_mod.input_port_type_id("DOES_NOT_EXIST")
        with pytest.raises(Exception):
            simple_mod.output_port_type_id("DOES_NOT_EXIST")

        builder.make_edge(source, simple_mod.input_port("input1"))
        builder.make_edge(source2, simple_mod.input_port("input2"))
        builder.make_edge(simple_mod.output_port("output1"), sink)
        builder.make_edge(simple_mod.output_port("output2"), sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("Initialization_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()


def test_py_module_as_source():

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

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

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSource_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()
    assert packet_count == 42


def test_py_module_as_sink():

    def gen_data():
        for i in range(0, 43):
            yield True
            global packet_count
            packet_count += 1

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        source = builder.make_source("source", gen_data())
        sink_mod = builder.load_module("SinkModule", "srf_unittest", "ModuleSinkTest_mod1", {})

        builder.make_edge(source, sink_mod.input_port("sink"))

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleAsSink_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 43


def test_py_module_chaining():

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(input):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(input))

        def on_error():
            pass

        def on_complete():
            pass

        config = {"source_count": 42}

        source_mod = builder.load_module("SourceModule", "srf_unittest", "ModuleChainingTest_mod1", config)
        configurable_mod = builder.load_module("ConfigurableModule", "srf_unittest", "ModuleEndToEndTest_mod2", {})
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(source_mod.output_port("source"), configurable_mod.input_port("configurable_input_a"))
        builder.make_edge(configurable_mod.output_port("configurable_output_x"), sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleChaining_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 42


def test_py_module_nesting():

    def gen_data():
        for i in range(0, 43):
            yield True
            global packet_count
            packet_count += 1

    def init_wrapper(builder: srf.Builder):
        global packet_count
        packet_count = 0

        def on_next(input):
            global packet_count
            packet_count += 1
            logging.info("Sinking {}".format(input))

        def on_error():
            pass

        def on_complete():
            pass

        nested_mod = builder.load_module("NestedModule", "srf_unittest", "ModuleNestingTest_mod1", {})
        nested_sink = builder.make_sink("nested_sink", on_next, on_error, on_complete)

        builder.make_edge(nested_mod.output_port("nested_module_output"), nested_sink)

    pipeline = srf.Pipeline()
    pipeline.make_segment("ModuleNesting_Segment", init_wrapper)

    options = srf.Options()
    options.topology.user_cpuset = "0-1"

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)
    executor.start()
    executor.join()

    assert packet_count == 4


if (__name__ in ("__main__", )):
    test_py_end_to_end()
    test_py_module_as_source()
    test_py_module_as_sink()
    test_py_module_chaining()
    test_py_module_nesting()
    test_py_constructor()
    test_py_module_initialization()
