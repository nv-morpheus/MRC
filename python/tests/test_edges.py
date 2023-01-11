# SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def add_source(seg: mrc.Builder, is_cpp: bool, data_type: type, is_component: bool):
    global node_counts, expected_node_counts

    prefix = "SourceComponent" if is_component else "Source"
    node_name = prefix + data_type.__name__

    expected_node_counts.update({
        f"{node_name}.on_next": 5,
        f"{node_name}.on_error": 0,
        f"{node_name}.on_completed": 1,
    })

    if (is_cpp):
        return getattr(m, node_name)(seg, node_name, node_counts)
    else:
        init_node_counter(f"{node_name}.on_next")
        init_node_counter(f"{node_name}.on_error")
        init_node_counter(f"{node_name}.on_completed")

        def source_fn():
            for _ in range(5):
                increment_node_counter(f"{node_name}.on_next")
                yield data_type()
            increment_node_counter(f"{node_name}.on_completed")

        if (is_component):
            return seg.make_source_component(node_name, source_fn())
        else:
            return seg.make_source(node_name, source_fn())


def add_sink(seg: mrc.Builder, upstream: mrc.SegmentObject, is_cpp: bool, data_type: type, is_component: bool):
    global node_counts, expected_node_counts

    prefix = "SinkComponent" if is_component else "Sink"
    node_name = prefix + data_type.__name__

    expected_node_counts.update({
        f"{node_name}.on_next": 5,
        f"{node_name}.on_error": 0,
        f"{node_name}.on_completed": 1,
    })

    sink = None

    if (is_cpp):
        sink = getattr(m, node_name)(seg, node_name, node_counts)
    else:
        init_node_counter(f"{node_name}.on_next")
        init_node_counter(f"{node_name}.on_error")
        init_node_counter(f"{node_name}.on_completed")

        def on_next_sink(x: int):
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


def add_cpp_source_component_base(seg: mrc.Builder):
    global node_counts
    node = m.SourceComponentBase(seg, "source_component_base", node_counts)

    return node


def add_cpp_source_component_derived(seg: mrc.Builder):
    node = m.SourceComponentDerivedA(seg, "cpp_source_component_derived")

    return node


def add_py_node_component_base(seg: mrc.Builder, upstream):

    def on_next_node(x: int):
        increment_node_counter("node_component_base.on_next")

        return x

    init_node_counter("node_component_base.on_next")

    node = seg.make_node_component("node_component_base", ops.map(on_next_node))

    seg.make_edge(upstream, node)

    return node


def add_cpp_sink_base(seg: mrc.Builder, upstream):

    node = m.SinkBase(seg, "sink_base")

    seg.make_edge(upstream, node)

    return node


def add_py_sink_component_base(seg: mrc.Builder, upstream):

    def on_next_sink(x: int):
        increment_node_counter("sink_component_base.on_next")

    def on_error_sink(err):
        increment_node_counter("sink_component_base.on_error")

    def on_completed_sink():
        increment_node_counter("sink_component_base.on_completed")

    node = seg.make_sink_component("sink_component_base", on_next_sink, on_error_sink, on_completed_sink)

    init_node_counter("sink_component_base.on_next")
    init_node_counter("sink_component_base.on_error")
    init_node_counter("sink_component_base.on_completed")

    seg.make_edge(upstream, node)

    return node


def test_connect_cpp_edges():

    def segment_init(seg: mrc.Builder):
        source = m.SourceDerivedB(seg, "source")

        node = m.NodeBase(seg, "node")
        seg.make_edge(source, node)

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(node, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_cpp_to_cpp_same():

    def segment_init(seg: mrc.Builder):
        source = m.SourceDerivedB(seg, "source")

        node = m.NodeBase(seg, "node")
        seg.make_edge(source, node)

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(node, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_cpp_to_py_same():

    def segment_init(seg: mrc.Builder):
        source = m.SourceDerivedB(seg, "source")

        def on_next(x: m.Base):
            pass

        def on_error(e):
            pass

        def on_complete():
            pass

        sink = seg.make_sink("sink", on_next, on_error, on_complete)
        seg.make_edge(source, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_py_to_cpp_same():

    def segment_init(seg: mrc.Builder):

        def source_fn():
            yield m.DerivedB()
            yield m.DerivedB()
            yield m.DerivedB()

        source = seg.make_source("source", source_fn())

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(source, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_wrapper():
    on_next_count = 0

    def segment_init(seg: mrc.Builder):

        def create_source():
            yield 1
            yield 2
            yield 3
            yield 4

        source = seg.make_source("source", create_source())
        # source = m.SourcePyHolder(seg, "source")

        node = m.NodePyHolder(seg, "node")
        seg.make_edge(source, node)

        def on_next(x: int):
            nonlocal on_next_count

            on_next_count += 1

        def on_error(e):
            pass

        def on_complete():
            pass

        sink = seg.make_sink("sink", on_next, on_error, on_complete)
        seg.make_edge(node, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert on_next_count == 4


def test_edge_wrapper_component():
    on_next_count = 0

    def segment_init(seg: mrc.Builder):

        def create_source():
            yield 1
            yield 2
            yield 3
            yield 4

        source = seg.make_source("source", create_source())

        # source = m.SourcePyHolder(seg, "source")

        def on_next(x: int):
            nonlocal on_next_count
            print("Got: {}".format(type(x)))

            on_next_count += 1

        def on_error(e):
            pass

        def on_complete():
            print("Complete")

        sink = seg.make_sink_component("sink_component", on_next, on_error, on_complete)
        seg.make_edge(source, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert on_next_count == 4


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


def test_broadcast_cpp_to_cpp_same():

    def segment_init(seg: mrc.Builder):
        source = m.SourceDerivedB(seg, "source")

        broadcast = mrc.core.node.Broadcast(seg, "broadcast")

        sink = m.SinkDerivedB(seg, "sink")

        seg.make_edge(source, broadcast)
        seg.make_edge(broadcast, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_broadcast_cpp_to_cpp_different():

    def segment_init(seg: mrc.Builder):
        source = m.SourceDerivedB(seg, "source")

        broadcast = mrc.core.node.Broadcast(seg, "broadcast")

        sink = m.SinkBase(seg, "sink")

        seg.make_edge(source, broadcast)
        seg.make_edge(broadcast, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_broadcast_cpp_to_cpp_multi():

    def segment_init(seg: mrc.Builder):
        source_derived = m.SourceDerivedB(seg, "source_derived")
        source_base = m.SourceBase(seg, "source_base")

        broadcast = mrc.core.node.Broadcast(seg, "broadcast")

        sink_base = m.SinkBase(seg, "sink_base")
        sink_derived = m.SinkDerivedB(seg, "sink_derived")

        seg.make_edge(source_derived, broadcast)
        seg.make_edge(source_base, broadcast)

        seg.make_edge(broadcast, sink_base)
        seg.make_edge(broadcast, sink_derived)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_source_cpp_to_sink_component_cpp():

    def segment_init(seg: mrc.Builder):
        source_base = m.SourceBase(seg, "source_base")

        sink_base = m.SinkComponentBase(seg, "sink_base")

        seg.make_edge(source_base, sink_base)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_source_cpp_to_sink_component_py():
    on_next_count = 0

    def segment_init(seg: mrc.Builder):

        source_base = m.SourceBase(seg, "source_base")

        def on_next(x: int):
            nonlocal on_next_count
            print("Got: {}".format(type(x)))

            on_next_count += 1

        def on_error(e):
            pass

        def on_complete():
            print("Complete")

        sink = seg.make_sink_component("sink_component", on_next, on_error, on_complete)

        seg.make_edge(source_base, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert on_next_count == 5


@pytest.mark.parametrize("source_component,sink_component",
                         [
                             pytest.param(False, False, id="runnable-runnable"),
                             pytest.param(True, False, id="component-runnable"),
                             pytest.param(False, True, id="runnable-component"),
                             pytest.param(True, True, id="component-component", marks=pytest.mark.xfail)
                         ])
@pytest.mark.parametrize("source_cpp", [True, False], ids=["source_cpp", "source_py"])
@pytest.mark.parametrize("sink_cpp", [True, False], ids=["sink_cpp", "sink_py"])
def test_source_base_to_sink_base(run_segment,
                                  source_component: bool,
                                  source_cpp: bool,
                                  sink_component: bool,
                                  sink_cpp: bool):

    def segment_init(seg: mrc.Builder):

        source = add_source(seg, is_cpp=source_cpp, data_type=m.Base, is_component=source_component)
        add_sink(seg, source, is_cpp=sink_cpp, data_type=m.Base, is_component=sink_component)

    results = run_segment(segment_init)

    assert results == expected_node_counts


def test_cpp_source_base_to_py_sink_component(run_segment):

    def segment_init(seg: mrc.Builder):

        source_base = add_cpp_source_base(seg)
        add_py_sink_component_base(seg, source_base)

    results = run_segment(segment_init)

    assert results == {
        "sink_component_base.on_next": 5,
        "sink_component_base.on_error": 0,
        "sink_component_base.on_completed": 1,
    }


def test_cpp_source_base_to_py_node_component_to_py_sink_component(run_segment):

    def segment_init(seg: mrc.Builder):

        source_base = add_cpp_source_base(seg)
        node_component = add_py_node_component_base(seg, source_base)
        add_py_sink_component_base(seg, node_component)

    results = run_segment(segment_init)

    assert results == {
        "node_component_base.on_next": 5,
        "sink_component_base.on_next": 5,
        "sink_component_base.on_error": 0,
        "sink_component_base.on_completed": 1,
    }


def test_cpp_source_component_base_to_cpp_sink_base(run_segment):

    def segment_init(seg: mrc.Builder):

        source_base = add_cpp_source_component_base(seg)
        add_cpp_sink_base(seg, source_base)

    results = run_segment(segment_init)

    assert results == {
        "source_component_base.on_next": 5,
        "source_component_base.on_complete": 1,
    }


def test_cpp_source_component_derived_to_cpp_sink_base(run_segment):

    def segment_init(seg: mrc.Builder):

        source_base = add_cpp_source_component_derived(seg)
        add_cpp_sink_base(seg, source_base)

    results = run_segment(segment_init)

    assert results == {}


if (__name__ == "__main__"):
    test_connect_cpp_edges()
    test_edge_cpp_to_cpp_same()
    test_edge_cpp_to_py_same()
    test_edge_py_to_cpp_same()
    test_edge_wrapper()
