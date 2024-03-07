# SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import itertools
import typing

import pytest

import mrc
import mrc.core.node
import mrc.core.operators as ops
import mrc.tests.test_edges_cpp as m


@pytest.fixture
def ex_runner():

    def run_exec(segment_init):
        pipeline = mrc.Pipeline()

        pipeline.make_segment("my_seg", segment_init)

        options = mrc.Options()

        # Set to 1 thread
        options.topology.user_cpuset = "0-0"

        executor = mrc.Executor(options)

        executor.register_pipeline(pipeline)

        executor.start()

        executor.join()

    return run_exec


@pytest.fixture
def run_segment(ex_runner):

    global node_counts, expected_node_counts

    def run(segment_fn):

        global node_counts, expected_node_counts

        # Reset node_counts, just to be sure
        node_counts = {}
        expected_node_counts = {}

        # Run the pipeline
        ex_runner(segment_fn)

        # Save the node counts and reset
        actual = node_counts

        return actual

    yield run

    # Reset after just to be sure
    node_counts = {}
    expected_node_counts = {}


def producer(to_produce):

    for x in to_produce:
        yield x


global node_counts
node_counts = {}

global expected_node_counts
expected_node_counts = {}


def init_node_counter(name: str):
    if (name not in node_counts):
        node_counts[name] = 0


def increment_node_counter(name: str):
    global node_counts

    init_node_counter(name)

    node_counts[name] += 1


def assert_node_counts(actual: dict, expected: dict):
    pass


def add_source(seg: mrc.Builder,
               is_cpp: bool,
               data_type: type,
               is_component: bool,
               suffix: str = "",
               msg_count: int = 5):
    global node_counts, expected_node_counts

    prefix = "SourceComponent" if is_component else "Source"
    node_type = prefix + data_type.__name__
    node_name = node_type + suffix

    expected_node_counts.update({
        f"{node_name}.on_next": msg_count,
        f"{node_name}.on_error": 0,
        f"{node_name}.on_completed": 1,
    })

    if (is_cpp):
        return getattr(m, node_type)(seg, node_name, node_counts, msg_count=msg_count)
    else:
        init_node_counter(f"{node_name}.on_next")
        init_node_counter(f"{node_name}.on_error")
        init_node_counter(f"{node_name}.on_completed")

        def source_fn():
            for _ in range(msg_count):
                increment_node_counter(f"{node_name}.on_next")
                yield data_type()
            increment_node_counter(f"{node_name}.on_completed")

        if (is_component):
            return seg.make_source_component(node_name, source_fn())
        else:
            return seg.make_source(node_name, source_fn())


def add_node(seg: mrc.Builder,
             upstream: mrc.SegmentObject,
             is_cpp: bool,
             data_type: type,
             is_component: bool,
             msg_count: int = 5):
    global node_counts, expected_node_counts

    prefix = "NodeComponent" if is_component else "Node"
    node_name = prefix + data_type.__name__

    expected_node_counts.update({
        f"{node_name}.on_next": msg_count,
        f"{node_name}.on_error": 0,
        f"{node_name}.on_completed": 1,
    })

    node = None

    if (is_cpp):
        node = getattr(m, node_name)(seg, node_name, node_counts, msg_count=msg_count)
    else:
        init_node_counter(f"{node_name}.on_next")
        init_node_counter(f"{node_name}.on_error")
        init_node_counter(f"{node_name}.on_completed")

        def on_next(x):
            assert isinstance(x, data_type)

            increment_node_counter(f"{node_name}.on_next")
            return x

        def on_completed():
            increment_node_counter(f"{node_name}.on_completed")

        if (is_component):
            node = seg.make_node_component(node_name, ops.map(on_next), ops.on_completed(on_completed))
        else:
            node = seg.make_node(node_name, ops.map(on_next), ops.on_completed(on_completed))

    seg.make_edge(upstream, node)

    return node


def add_sink(seg: mrc.Builder,
             upstream: mrc.SegmentObject,
             is_cpp: bool,
             data_type: type,
             is_component: bool,
             suffix: str = "",
             count: int = 5,
             expected_vals_fn: typing.Callable[[typing.Dict[str, int]], typing.Dict[str, int]] = None):
    global node_counts, expected_node_counts

    prefix = "SinkComponent" if is_component else "Sink"
    node_type = prefix + data_type.__name__
    node_name = node_type + suffix

    expected_orig = {
        "on_next": count,
        "on_error": 0,
        "on_completed": 1,
    }

    if (expected_vals_fn is not None):
        expected_orig = expected_vals_fn(expected_orig)

    expected_node_counts.update({f"{node_name}.{k}": v for k, v in expected_orig.items()})

    sink = None

    if (is_cpp):
        sink = getattr(m, node_type)(seg, node_name, node_counts)
    else:
        init_node_counter(f"{node_name}.on_next")
        init_node_counter(f"{node_name}.on_error")
        init_node_counter(f"{node_name}.on_completed")

        def on_next_sink(x):
            assert isinstance(x, data_type)
            increment_node_counter(f"{node_name}.on_next")

        def on_error_sink(err):
            increment_node_counter(f"{node_name}.on_error")

        def on_completed_sink():
            increment_node_counter(f"{node_name}.on_completed")

        if (is_component):
            sink = seg.make_sink_component(node_name, on_next_sink, on_error_sink, on_completed_sink)
        else:
            sink = seg.make_sink(node_name, on_next_sink, on_error_sink, on_completed_sink)

    seg.make_edge(upstream, sink)

    return sink


def add_broadcast(seg: mrc.Builder, *upstream: mrc.SegmentObject):

    node = mrc.core.node.Broadcast(seg, "Broadcast")

    for u in upstream:
        seg.make_edge(u, node)

    return node


def add_round_robin_router(seg: mrc.Builder, *upstream: mrc.SegmentObject):

    node = mrc.core.node.RoundRobinRouter(seg, "RoundRobinRouter")

    for u in upstream:
        seg.make_edge(u, node)

    return node


# THIS TEST IS CAUSING ISSUES WHEN RUNNING ALL TESTS TOGETHER

# @dataclasses.dataclass
# class MyCustomClass:
#     value: int
#     name: str

# def test_multi_segment():

#     def segment_source(seg: mrc.Builder):
#         # Use a generator function as the source
#         def source_gen():
#             for i in range(5):
#                 yield MyCustomClass(i, "Instance-{}".format(i))
#                 # yield m.DerivedA()

#         def source_untyped():
#             for i in range(5):
#                 yield 1

#         # Create the source object
#         # source = seg.make_source("source", source_gen)
#         source = m.SourceDerivedB(seg, "source")
#         source.launch_options.pe_count = 1

#         egress = seg.get_egress("port1")
#         seg.make_edge(source, egress)

#         source2 = seg.make_source("source_untyped", source_untyped)
#         egress2 = seg.get_egress("port2")
#         seg.make_edge(source2, egress2)

#     def segment_sink(seg: mrc.Builder):
#         ingress = seg.get_ingress("port1")

#         # This method will get called each time the sink gets a value
#         def sink_on_next(x: MyCustomClass):
#             pass

#         def sink_on_next_untyped(input):
#             pass

#         def sink_on_error():
#             pass

#         def sink_on_complete():
#             pass

#         # Build the sink object
#         # sink = seg.make_sink("sink", sink_on_next, None, None)
#         sink = m.SinkBase(seg, "sink")

#         seg.make_edge(ingress, sink)

#         sink2 = seg.make_sink("sink_untyped", sink_on_next_untyped, sink_on_complete, sink_on_error)
#         ingress2 = seg.get_ingress("port2")
#         seg.make_edge(ingress2, sink2)

#     mrc.Config.default_channel_size = 4

#     # Create the pipeline object
#     pipeline = mrc.Pipeline()

#     # Create a segment
#     pipeline.make_segment("segment_source", [], [("port1", m.DerivedB), "port2"], segment_source)

#     pipeline.make_segment("segment_sink", [("port1", m.DerivedB), "port2"], [], segment_sink)

#     # Build executor options
#     options = mrc.Options()

#     # Set to 1 thread
#     options.topology.user_cpuset = "0-0"

#     # Create the executor
#     executor = mrc.Executor(options)

#     # Register pipeline to tell executor what to run
#     executor.register_pipeline(pipeline)

#     # This will start the pipeline and return immediately
#     executor.start()

#     # Wait for the pipeline to exit on its own
#     executor.join()


def gen_parameters(*args,
                   is_fail_fn: typing.Callable[[typing.Tuple], bool],
                   values: typing.Dict[str, typing.Any] = {
                       "com": True, "run": False
                   }):

    all_node_names = list(args)

    parameters = []

    for combo in itertools.product(values.keys(), repeat=len(all_node_names)):

        marks = ()

        combo_vals = tuple(values[x] for x in combo)

        if (is_fail_fn(combo_vals)):
            marks = (pytest.mark.xfail, )

        p = pytest.param(*combo_vals, id="-".join([f"{x[0]}_{x[1]}" for x in zip(args, combo)]), marks=marks)

        parameters.append(p)

    return parameters


@pytest.mark.parametrize("source_component,sink_component", gen_parameters("source", "sink", is_fail_fn=all))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink_cpp", [True, False], ids=["sink_cpp", "sink_py"])
@pytest.mark.parametrize("source_type,sink_type",
                         [
                             pytest.param(m.Base, m.Base, id="source_base-sink_base"),
                             pytest.param(m.Base, m.DerivedA, id="source_base-sink_derived", marks=pytest.mark.xfail),
                             pytest.param(m.DerivedA, m.Base, id="source_derived-sink_base"),
                             pytest.param(m.DerivedA, m.DerivedA, id="source_derived-sink_derived")
                         ])
def test_source_to_sink(run_segment,
                        source_component: bool,
                        sink_component: bool,
                        source_cpp: bool,
                        sink_cpp: bool,
                        source_type: type,
                        sink_type: type):

    def segment_init(seg: mrc.Builder):

        source = add_source(seg, is_cpp=source_cpp, data_type=source_type, is_component=source_component)
        add_sink(seg, source, is_cpp=sink_cpp, data_type=sink_type, is_component=sink_component)

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize("source_component,node_component,sink_component",
                         gen_parameters("source", "node", "sink", is_fail_fn=lambda c: c[0] and c[1]))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("node_cpp", [True, False], ids=["node_cpp", "node_py"])
@pytest.mark.parametrize("sink_cpp", [True, False], ids=["sink_cpp", "sink_py"])
def test_source_to_node_to_sink(run_segment,
                                source_component: bool,
                                node_component: bool,
                                sink_component: bool,
                                source_cpp: bool,
                                node_cpp: bool,
                                sink_cpp: bool):

    def segment_init(seg: mrc.Builder):

        source = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=source_component)
        node = add_node(seg, source, is_cpp=node_cpp, data_type=m.Base, is_component=node_component)
        add_sink(seg, node, is_cpp=sink_cpp, data_type=m.Base, is_component=sink_component)

    results = run_segment(segment_init)

    assert results == expected_node_counts


def fail_if_more_derived_type(combo: typing.Tuple):

    for prev_type, cur_type in zip(combo[0::1], combo[1::1]):
        if (not issubclass(prev_type, cur_type)):
            return True

    return False


@pytest.mark.parametrize("sink1_component,sink2_component",
                         gen_parameters("sink1", "sink2", is_fail_fn=lambda x: False))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink1_cpp", [True, False], ids=["sink1_cpp", "sink2_py"])
@pytest.mark.parametrize("sink2_cpp", [True, False], ids=["sink2_cpp", "sink2_py"])
@pytest.mark.parametrize(
    "source_type,sink1_type,sink2_type",
    gen_parameters("source",
                   "sink1",
                   "sink2",
                   is_fail_fn=fail_if_more_derived_type,
                   values={
                       "base": m.Base, "derived": m.DerivedA
                   }))
def test_source_to_broadcast_to_sinks(run_segment,
                                      sink1_component: bool,
                                      sink2_component: bool,
                                      source_cpp: bool,
                                      sink1_cpp: bool,
                                      sink2_cpp: bool,
                                      source_type: type,
                                      sink1_type: type,
                                      sink2_type: type):

    def segment_init(seg: mrc.Builder):

        source = add_source(seg, is_cpp=source_cpp, data_type=source_type, is_component=False)
        broadcast = add_broadcast(seg, source)
        add_sink(seg, broadcast, is_cpp=sink1_cpp, data_type=sink1_type, is_component=sink1_component, suffix="1")
        add_sink(seg, broadcast, is_cpp=sink2_cpp, data_type=sink2_type, is_component=sink2_component, suffix="2")

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize("sink1_component,sink2_component",
                         gen_parameters("sink1", "sink2", is_fail_fn=lambda x: False))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink1_cpp", [True, False], ids=["sink1_cpp", "sink1_py"])
@pytest.mark.parametrize("sink2_cpp", [True, False], ids=["sink2_cpp", "sink2_py"])
def test_multi_source_to_broadcast_to_multi_sink(run_segment,
                                                 sink1_component: bool,
                                                 sink2_component: bool,
                                                 source_cpp: bool,
                                                 sink1_cpp: bool,
                                                 sink2_cpp: bool):

    def double_on_next(exp):
        exp["on_next"] *= 2

        return exp

    def segment_init(seg: mrc.Builder):

        source1 = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=False, suffix="1")
        source2 = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=False, suffix="2")
        broadcast = add_broadcast(seg, source1, source2)
        add_sink(seg,
                 broadcast,
                 is_cpp=sink1_cpp,
                 data_type=m.Base,
                 is_component=sink1_component,
                 suffix="1",
                 expected_vals_fn=double_on_next)
        add_sink(seg,
                 broadcast,
                 is_cpp=sink2_cpp,
                 data_type=m.Base,
                 is_component=sink2_component,
                 suffix="2",
                 expected_vals_fn=double_on_next)

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize("sink1_component,sink2_component",
                         gen_parameters("sink1", "sink2", is_fail_fn=lambda x: False))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink1_cpp", [True, False], ids=["sink1_cpp", "sink2_py"])
@pytest.mark.parametrize("sink2_cpp", [True, False], ids=["sink2_cpp", "sink2_py"])
@pytest.mark.parametrize(
    "source_type,sink1_type,sink2_type",
    gen_parameters("source",
                   "sink1",
                   "sink2",
                   is_fail_fn=fail_if_more_derived_type,
                   values={
                       "base": m.Base, "derived": m.DerivedA
                   }))
def test_source_to_round_robin_router_to_sinks(run_segment,
                                               sink1_component: bool,
                                               sink2_component: bool,
                                               source_cpp: bool,
                                               sink1_cpp: bool,
                                               sink2_cpp: bool,
                                               source_type: type,
                                               sink1_type: type,
                                               sink2_type: type):

    def segment_init(seg: mrc.Builder):

        source = add_source(seg, is_cpp=source_cpp, data_type=source_type, is_component=False)
        broadcast = add_round_robin_router(seg, source)
        add_sink(seg,
                 broadcast,
                 is_cpp=sink1_cpp,
                 data_type=sink1_type,
                 is_component=sink1_component,
                 suffix="1",
                 count=3)
        add_sink(seg,
                 broadcast,
                 is_cpp=sink2_cpp,
                 data_type=sink2_type,
                 is_component=sink2_component,
                 suffix="2",
                 count=2)

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize("sink1_component,sink2_component",
                         gen_parameters("sink1", "sink2", is_fail_fn=lambda x: False))
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink1_cpp", [True, False], ids=["sink1_cpp", "sink1_py"])
@pytest.mark.parametrize("sink2_cpp", [True, False], ids=["sink2_cpp", "sink2_py"])
def test_multi_source_to_round_robin_router_to_multi_sink(run_segment,
                                                          sink1_component: bool,
                                                          sink2_component: bool,
                                                          source_cpp: bool,
                                                          sink1_cpp: bool,
                                                          sink2_cpp: bool):

    def segment_init(seg: mrc.Builder):

        source1 = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=False, suffix="1")
        source2 = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=False, suffix="2")
        broadcast = add_round_robin_router(seg, source1, source2)
        add_sink(seg, broadcast, is_cpp=sink1_cpp, data_type=m.Base, is_component=sink1_component, suffix="1")
        add_sink(seg, broadcast, is_cpp=sink2_cpp, data_type=m.Base, is_component=sink2_component, suffix="2")

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize(
    "source_type", gen_parameters("source", is_fail_fn=lambda _: False, values={
        "base": m.Base, "derived": m.DerivedA
    }))
def test_source_to_null(run_segment, source_cpp: bool, source_type: type):

    def segment_init(seg: mrc.Builder):

        # Add a large enough count to fill a buffered channel
        add_source(seg, is_cpp=source_cpp, data_type=source_type, is_component=False, msg_count=500)

    results = run_segment(segment_init)

    assert results == expected_node_counts


@pytest.mark.parametrize(
    "source_cpp,node_cpp",
    gen_parameters("source", "node", is_fail_fn=lambda _: False, values={
        "cpp": True, "py": False
    }))
@pytest.mark.parametrize(
    "source_type,node_type",
    gen_parameters("source",
                   "node",
                   is_fail_fn=fail_if_more_derived_type,
                   values={
                       "base": m.Base, "derived": m.DerivedA
                   }))
@pytest.mark.parametrize(
    "source_component,node_component",
    gen_parameters("source", "node", is_fail_fn=lambda x: x[0] and x[1], values={
        "run": False, "com": True
    }))
def test_source_to_node_to_null(run_segment,
                                source_cpp: bool,
                                node_cpp: bool,
                                source_type: type,
                                node_type: type,
                                source_component: bool,
                                node_component: bool):

    def segment_init(seg: mrc.Builder):

        # Add a large enough count to fill a buffered channel
        source = add_source(seg, is_cpp=source_cpp, data_type=source_type, is_component=source_component, msg_count=500)
        add_node(seg, source, is_cpp=node_cpp, data_type=node_type, is_component=node_component, msg_count=500)

    results = run_segment(segment_init)

    assert results == expected_node_counts
