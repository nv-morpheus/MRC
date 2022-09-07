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

# import gc
# import pathlib
# import sys
# from functools import partial

# import numpy as np
# import pytest

import srf
import srf.tests.test_edges_cpp as m

# from srf.core.options import PlacementStrategy

# whereami = pathlib.Path(__file__).parent.resolve()

# def map_upper(x: str):
#     return x.upper()

# def capture_sink_on_next(x, tracing_dict):
#     # s = f"(python)Sinking {str(x)} : {sys.getrefcount(x)}"
#     # print(s)
#     tracing_dict["inputs"].append(x)
#     tracing_dict["on_next"] += 1

# def capture_sink_on_error(x, tracing_dict):
#     tracing_dict["inputs"].append(x)
#     tracing_dict["on_error"] += 1

# def capture_sink_on_completed(tracing_dict):
#     # print("(python)Sink.on_completed")
#     tracing_dict["on_completed"] += 1

# ### Pure Python example
# def string_producer():
#     lines = [
#         "This is a test 1",
#         "This is a test 2",
#         "This is a test 3",
#     ]

#     for line in lines:
#         yield line

#     return

# STRING_READER_INPUTS = []
# with open(f"{whereami}/string_reader_input.txt") as reader:
#     data = reader.read()
#     STRING_READER_INPUTS.extend(data.splitlines())

# def file_reader():
#     with open(f"{whereami}/string_reader_input.txt") as reader:
#         data = reader.read()

#     for line in data.splitlines():
#         print(f"Yielding {line}")
#         yield line

#     return

# def vector_file_producer():
#     with open(f"{whereami}/string_reader_input.txt") as reader:
#         data = reader.read()

#     i = 0
#     for line in data.splitlines():
#         item = srf.VectorWrapperThing(line)
#         print(f"Item refcount before yield: {sys.getrefcount(item)}")
#         print(f"Item referrers before yield: {gc.get_referrers(item)}")
#         yield line
#         i += 1
#         if (i >= 10):
#             break

#     return

# TRACING_DICT = {}

# def map_double_float(f):
#     rval = f * 2.212
#     return rval

# def segment_init_homogesrfus_string_usage(seg: srf.Builder):
#     on_next = partial(capture_sink_on_next, tracing_dict=TRACING_DICT)
#     on_error = partial(capture_sink_on_error, tracing_dict=TRACING_DICT)
#     on_completed = partial(capture_sink_on_completed, tracing_dict=TRACING_DICT)

#     ### Pure Python example
#     pynode_string_source = seg.make_source("pynode_string", string_producer())
#     pynode_to_upper = seg.make_node("upper_node", map_upper)
#     seg.make_edge(pynode_string_source, pynode_to_upper)

#     py_string_sink = seg.make_sink("py_sink", on_next, on_error, on_completed)
#     seg.make_edge(pynode_to_upper, py_string_sink)
#     ### Pure Python example

# def segment_init_cxx_string_source(seg: srf.Builder):
#     on_next = partial(capture_sink_on_next, tracing_dict=TRACING_DICT)
#     on_error = partial(capture_sink_on_error, tracing_dict=TRACING_DICT)
#     on_completed = partial(capture_sink_on_completed, tracing_dict=TRACING_DICT)

#     ### CXX std::string source input to pure python function chain
#     pynode_string_src = seg.make_file_reader("pynode_string_src", f"{whereami}/string_reader_input.txt")
#     pynode_to_upper_1 = seg.make_node("pynode_to_upper_1", map_upper)
#     seg.make_cxx2py_edge_adapter(pynode_string_src, pynode_to_upper_1, np.str_)
#     pysink = seg.make_sink("my_pysink", on_next, on_error, on_completed)
#     seg.make_edge(pynode_to_upper_1, pysink)
#     ###

# def segment_init_heterogesrfus_string_usage(seg: srf.Builder):
#     on_next = partial(capture_sink_on_next, tracing_dict=TRACING_DICT)
#     on_error = partial(capture_sink_on_error, tracing_dict=TRACING_DICT)
#     on_completed = partial(capture_sink_on_completed, tracing_dict=TRACING_DICT)

#     ### python string source input to heterogenous python chain
#     pynode_str_source = seg.make_source("pynode_str_src", file_reader())
#     pynode_to_upper_1 = seg.make_node("pynode_to_upper_1", map_upper)
#     seg.make_edge(pynode_str_source, pynode_to_upper_1)
#     # Python string source linked to a python function to convert to upper case

#     node_string_passthrough = seg.debug_string_passthrough("node_string_passthrough")
#     seg.make_py2cxx_edge_adapter(pynode_to_upper_1, node_string_passthrough, np.str_)
#     # Convert Python string output to cxx std::string

#     pynode_to_upper_2 = seg.make_node("pynode_to_upper_2", map_upper)
#     seg.make_cxx2py_edge_adapter(node_string_passthrough, pynode_to_upper_2, np.str_)
#     # seg.make_edge(pynode_to_upper_1, pynode_to_upper_2, np.str_)
#     # Convert cxx std::string output to python string

#     py_string_sink = seg.make_sink("py_string_sink", on_next, on_error, on_completed)
#     seg.make_edge(pynode_to_upper_2, py_string_sink)
#     # Sink python string
#     ###
#     print("Hetero string usage graph constructed")

# def segment_initcpp_flatten(seg: srf.Builder):
#     on_next = partial(capture_sink_on_next, tracing_dict=TRACING_DICT)
#     on_error = partial(capture_sink_on_error, tracing_dict=TRACING_DICT)
#     on_completed = partial(capture_sink_on_completed, tracing_dict=TRACING_DICT)

#     pynode_slist_source = seg.make_source("pynode_slist_src", vector_file_producer())
#     node_flatten_list = seg.flatten_list("node_flatten_list")
#     seg.make_edge(pynode_slist_source, node_flatten_list)
#     # node_string_passthrough = seg.debug_string_passthrough("string_passthrough")
#     # seg.make_py2cxx_edge_adapter(node_flatten_list, node_string_passthrough, np.str_)
#     # seg.make_edge(node_flatten_list, node_string_passthrough)
#     # pynode_to_upper_2 = seg.make_node("pynode_to_upper_2", map_upper)
#     py_string_sink = seg.make_sink("py_string_sink", on_next, on_error, on_completed)
#     # seg.make_edge(pynode_to_upper_2, py_string_sink)
#     # seg.make_edge(node_string_passthrough, py_string_sink)
#     seg.make_cxx2py_edge_adapter(node_flatten_list, py_string_sink, np.str_)
#     ###

# def segment_init_heterogenous_double_usage(seg: srf.Builder):
#     on_next = partial(capture_sink_on_next, tracing_dict=TRACING_DICT)
#     on_error = partial(capture_sink_on_error, tracing_dict=TRACING_DICT)
#     on_completed = partial(capture_sink_on_completed, tracing_dict=TRACING_DICT)

#     ### CXX double source with heterogesrfus segment node composition
#     source_double = seg.debug_float_source("source_double", 10000)  # inject new double's into the segment
#     pynode_multiplier_1 = seg.make_node(
#         "pynode_multiplier_1", map_double_float)  # Python node that casts our value to a pyobject and doubles it
#     seg.make_cxx2py_edge_adapter(source_double, pynode_multiplier_1, np.float64)

#     pynode_multiplier_2 = seg.make_node("pynode_multiplier_2", map_double_float)
#     seg.make_edge(pynode_multiplier_1, pynode_multiplier_2)

#     node_double_passthrough = seg.debug_float_passthrough("node_double_passthrough")
#     seg.make_py2cxx_edge_adapter(pynode_multiplier_2, node_double_passthrough, np.float64)

#     pynode_multiplier_3 = seg.make_node("pynode_multiplier_3", map_double_float)
#     seg.make_cxx2py_edge_adapter(node_double_passthrough, pynode_multiplier_3, np.float64)

#     pysink_numeric = seg.make_sink("pysink_numeric", on_next, on_error, on_completed)
#     seg.make_edge(pynode_multiplier_3, pysink_numeric)

# def do_segment_test(name, init_func):
#     global TRACING_DICT
#     TRACING_DICT = {"inputs": [], "outputs": [], "on_next": 0, "on_error": 0, "on_completed": 0}

#     pipeline = srf.Pipeline()
#     pipeline.make_segment(name, init_func)

#     options = srf.Options()
#     options.placement.cpu_strategy = PlacementStrategy.PerMachine

#     executor = srf.Executor(options)
#     executor.register_pipeline(pipeline)

#     executor.start()
#     executor.join()

# #@pytest.mark.skip("Doesn't work yet.")
# @pytest.mark.xfail  # issue#161
# def test_list_flatten_test():
#     do_segment_test("list flatten", segment_initcpp_flatten)

# @pytest.mark.xfail  # issue#161
# def test_homogenous_string_usage():
#     global TRACING_DICT
#     do_segment_test("homogenous_string_segment", segment_init_homogesrfus_string_usage)
#     assert (TRACING_DICT['on_next'] == 3)
#     assert (TRACING_DICT['on_error'] == 0)
#     assert (TRACING_DICT['on_completed'] == 1)

#     expected_inputs = ["THIS IS A TEST 1", "THIS IS A TEST 2", "THIS IS A TEST 3"]
#     for expected, actual in zip(expected_inputs, TRACING_DICT["inputs"]):
#         assert (expected == actual)

# @pytest.mark.xfail  # issue#161
# def test_cxx_string_source_to_python_chain():
#     global TRACING_DICT
#     do_segment_test("cxx_string_source_segment", segment_init_cxx_string_source)

#     assert (TRACING_DICT['on_next'] == 220)
#     assert (TRACING_DICT['on_error'] == 0)
#     assert (TRACING_DICT['on_completed'] == 1)

#     for expected, actual in zip(STRING_READER_INPUTS, TRACING_DICT["inputs"]):
#         assert (expected.strip().upper() == actual)

# @pytest.mark.xfail  # issue#161
# def test_heterogenous_string_usage():
#     global TRACING_DICT
#     do_segment_test("heterogenous_string_segment", segment_init_heterogesrfus_string_usage)

#     assert (TRACING_DICT['on_next'] == 220)
#     assert (TRACING_DICT['on_error'] == 0)
#     assert (TRACING_DICT['on_completed'] == 1)

#     for expected, actual in zip(STRING_READER_INPUTS, TRACING_DICT["inputs"]):
#         if (expected.strip().upper() != actual):
#             print(expected.strip().upper())
#             print(actual)
#         assert (expected.strip().upper() == actual)

# @pytest.mark.xfail  # issue#161
# def test_heterogenous_double_pipeline():
#     global TRACING_DICT
#     do_segment_test("heterogesrfus_double_segment", segment_init_heterogenous_double_usage)

#     assert (TRACING_DICT['on_next'] == 10000)
#     assert (TRACING_DICT['on_error'] == 0)
#     assert (TRACING_DICT['on_completed'] == 1)

#     for actual in TRACING_DICT["inputs"]:
#         assert (np.isclose(actual, 34.002060877715685))


def test_pipeline_creation_noports():

    def init(seg):
        pass

    pipe = srf.Pipeline()
    pipe.make_segment("TestSegment1", init)
    pipe.make_segment("TestSegment2", [], [], init)


"""
Test that the python bindings for segment creation with ingress and/or egress ports works as expected.
Since this is a runtime operator and all IngressCount X EgressCount pairs will generate a new class instance that is
explciitly defined, we check all of them.
"""


def test_dynamic_port_creation_good():

    def init(builder):
        pass

    ingress = [f"{chr(i)}" for i in range(65, 76)]
    egress = [f"{chr(i)}" for i in range(97, 108)]

    for i in range(len(ingress)):
        for j in range(len(egress)):
            pipe = srf.Pipeline()
            pipe.make_segment("DynamicPortTestSegment", ingress[0:i], egress[0:j], init)


def test_dynamic_port_creation_bad():

    def init(builder):
        pass

    ingress = [(f"{chr(i)}", 'c') for i in range(65, 76)]
    egress = [(f"{chr(i)}", 12) for i in range(97, 108)]

    pipe = srf.Pipeline()
    try:
        pipe.make_segment("DynamicPortTestSegmentIngress", ingress, [], init)
        assert (False)
    except Exception as e:
        print(e)
        pass

    try:
        pipe.make_segment("DynamicPortTestSegmentEgress", [], egress, init)
        assert (False)
    except Exception as e:
        print(e)
        pass


def test_ingress_egress_custom_type_construction():

    def gen_data():
        yield 1
        yield 2
        yield 3

    def init1(builder: srf.Builder):
        source = builder.make_source("source", gen_data)
        egress = builder.get_egress("b")

        builder.make_edge(source, egress)

    def init2(builder: srf.Builder):

        def on_next(input):
            pass

        def on_error():
            pass

        def on_complete():
            pass

        ingress = builder.get_ingress("b")
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(ingress, sink)

    pipe = srf.Pipeline()

    # Create segments with various combinations of type and untyped ports
    pipe.make_segment("TestSegment1", [("c", m.DerivedA), "c21", ("c31", int, False), "c41"],
                      ["d11", ("d", m.DerivedA)],
                      init1)
    pipe.make_segment("TestSegment2", [("a", m.DerivedB), "a22", "e32"],
                      [("b21", dict, False), "b22", ("b", m.DerivedB)],
                      init2)
    pipe.make_segment("TestSegment3", [("e", m.Base), ("e23", list, False), "e33"], ["f11", ("f", m.Base), "f13"],
                      init1)


def test_dynamic_port_get_ingress_egress():

    def gen_data():
        yield 1
        yield 2
        yield 3

    def init1(builder: srf.Builder):
        source = builder.make_source("source", gen_data)
        egress = builder.get_egress("b")

        builder.make_edge(source, egress)

    def init2(builder: srf.Builder):

        def on_next(input):
            pass

        def on_error():
            pass

        def on_complete():
            pass

        ingress = builder.get_ingress("b")
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(ingress, sink)

    pipe = srf.Pipeline()

    pipe.make_segment("TestSegment11", [], ["b"], init1)
    pipe.make_segment("TestSegment22", ["b"], [], init2)

    options = srf.Options()

    executor = srf.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()


def test_dynamic_port_with_type_get_ingress_egress():

    def gen_data():
        yield 1
        yield 2
        yield 3

    def init1(builder: srf.Builder):
        source = builder.make_source("source", gen_data)
        egress = builder.get_egress("b")

        builder.make_edge(source, egress)

    def init2(builder: srf.Builder):

        def on_next(input):
            pass

        def on_error():
            pass

        def on_complete():
            pass

        ingress = builder.get_ingress("b")
        sink = builder.make_sink("sink", on_next, on_error, on_complete)

        builder.make_edge(ingress, sink)

    pipe = srf.Pipeline()

    pipe.make_segment("TestSegment11", [], [("b", int, False)], init1)
    pipe.make_segment("TestSegment22", [("b", int, False)], [], init2)

    options = srf.Options()

    executor = srf.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()


if (__name__ in ("__main__", )):
    test_dynamic_port_creation_good()
    test_dynamic_port_creation_bad()
    test_ingress_egress_custom_type_construction()
    test_dynamic_port_get_ingress_egress()
    test_dynamic_port_with_type_get_ingress_egress()
