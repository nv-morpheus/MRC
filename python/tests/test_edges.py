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

import dataclasses

import mrc
import mrc.core.node
import mrc.core.operators as ops
import mrc.tests.test_edges_cpp as m


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


@dataclasses.dataclass
class MyCustomClass:
    value: int
    name: str


def test_multi_segment():

    def segment_source(seg: mrc.Builder):
        # Use a generator function as the source
        def source_gen():
            for i in range(5):
                yield MyCustomClass(i, "Instance-{}".format(i))
                # yield m.DerivedA()

        def source_untyped():
            for i in range(5):
                yield 1

        # Create the source object
        # source = seg.make_source("source", source_gen)
        source = m.SourceDerivedB(seg, "source")
        source.launch_options.pe_count = 1

        egress = seg.get_egress("port1")
        seg.make_edge(source, egress)

        source2 = seg.make_source("source_untyped", source_untyped)
        egress2 = seg.get_egress("port2")
        seg.make_edge(source2, egress2)

    def segment_sink(seg: mrc.Builder):
        ingress = seg.get_ingress("port1")

        # This method will get called each time the sink gets a value
        def sink_on_next(x: MyCustomClass):
            pass

        def sink_on_next_untyped(input):
            pass

        def sink_on_error():
            pass

        def sink_on_complete():
            pass

        # Build the sink object
        # sink = seg.make_sink("sink", sink_on_next, None, None)
        sink = m.SinkBase(seg, "sink")

        seg.make_edge(ingress, sink)

        sink2 = seg.make_sink("sink_untyped", sink_on_next_untyped, sink_on_complete, sink_on_error)
        ingress2 = seg.get_ingress("port2")
        seg.make_edge(ingress2, sink2)

    mrc.Config.default_channel_size = 4

    # Create the pipeline object
    pipeline = mrc.Pipeline()

    # Create a segment
    pipeline.make_segment("segment_source", [], [("port1", m.DerivedB), "port2"], segment_source)

    pipeline.make_segment("segment_sink", [("port1", m.DerivedB), "port2"], [], segment_sink)

    # Build executor options
    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    # Create the executor
    executor = mrc.Executor(options)

    # Register pipeline to tell executor what to run
    executor.register_pipeline(pipeline)

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()


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


def test_source_cpp_to_node_component_py_to_sink_component_py():
    on_next_count_node = 0
    on_next_count_sink = 0

    def segment_init(seg: mrc.Builder):

        source_base = m.SourceBase(seg, "source_base")

        def on_next_node(x: int):
            nonlocal on_next_count_node

            on_next_count_node += 1

            return x

        node = seg.make_node_component("node_component", ops.map(on_next_node))

        def on_next_sink(x: int):
            nonlocal on_next_count_sink

            on_next_count_sink += 1

        def on_error_sink(err):
            pass

        def on_completed_sink():
            print("Completed")

        sink = seg.make_sink_component("sink_component", on_next_sink, on_error_sink, on_completed_sink)

        seg.make_edge(source_base, node)
        seg.make_edge(node, sink)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-1"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert on_next_count_sink == 5


if (__name__ == "__main__"):
    test_connect_cpp_edges()
    test_edge_cpp_to_cpp_same()
    test_edge_cpp_to_py_same()
    test_edge_py_to_cpp_same()
    test_edge_wrapper()
    test_multi_segment()
