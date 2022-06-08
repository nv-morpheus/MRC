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

import srf
import srf.tests.test_edges_cpp as m


def test_connect_cpp_edges():
    def segment_init(seg: srf.Builder):
        source = m.SourceDerivedB(seg, "source")

        node = m.NodeBase(seg, "node")
        seg.make_edge(source, node)

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(node, sink)

    pipeline = srf.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_cpp_to_cpp_same():
    def segment_init(seg: srf.Builder):
        source = m.SourceDerivedB(seg, "source")

        node = m.NodeBase(seg, "node")
        seg.make_edge(source, node)

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(node, sink)

    pipeline = srf.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_cpp_to_py_same():
    def segment_init(seg: srf.Builder):
        source = m.SourceDerivedB(seg, "source")

        def on_next(x: m.Base):
            print("Got: {}".format(type(x)))

        def on_error(e):
            pass

        def on_complete():
            print("Complete")

        sink = seg.make_sink("sink", on_next, on_error, on_complete)
        seg.make_edge(source, sink)

    pipeline = srf.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_py_to_cpp_same():
    def segment_init(seg: srf.Builder):
        def source_fn():
            yield m.DerivedB()
            yield m.DerivedB()
            yield m.DerivedB()

        source = seg.make_source("source", source_fn())

        sink = m.SinkBase(seg, "sink")
        seg.make_edge(source, sink)

    pipeline = srf.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


def test_edge_wrapper():

    on_next_count = 0

    def segment_init(seg: srf.Builder):
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
            print("Got: {}".format(type(x)))

            on_next_count += 1

        def on_error(e):
            pass

        def on_complete():
            print("Complete")

        sink = seg.make_sink("sink", on_next, on_error, on_complete)
        seg.make_edge(node, sink)

    pipeline = srf.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    executor = srf.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert on_next_count == 4


if (__name__ == "__main__"):
    test_edge_py_to_cpp_same()
