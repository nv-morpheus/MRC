# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import mrc

VERSION = [int(cmpt) for cmpt in mrc.__version__.split(".")]


def test_mirror_tap_init():
    mirror_tap_one = mrc.MirrorTap("test_mirror_tap")
    mirror_tap_two = mrc.MirrorTap("test_mirror_tap", {"test": "test"})

    assert (mirror_tap_one is not None)
    assert (mirror_tap_two is not None)


def test_single_pipeline_tap_and_buffer():
    global packet_count, packets_main, packets_mirrored
    packet_count, packets_main, packets_mirrored = 10000, 0, 0

    global test_name
    test_name = "test_single_pipeline_tap_and_buffer"

    def gen_data_one():
        global packet_count
        for i in range(packet_count):
            yield {"data": i}

    def init_wrapper_main(builder: mrc.Builder):

        def on_next_sink(input):
            global packets_main
            packets_main += 1

        def on_error():
            pass

        def on_complete():
            pass

        source = builder.make_source(test_name + "_main_source", gen_data_one())
        sink = builder.make_sink(test_name + "_main_sink", on_next_sink, on_error, on_complete)

        builder.make_edge(source, sink)

    def init_wrapper_mirrored(builder: mrc.Builder):

        def on_next_sink(input):
            global packets_mirrored
            packets_mirrored += 1

        def on_error():
            pass

        def on_complete():
            pass

        builder.make_sink(test_name + "_mirror_sink", on_next_sink, on_error, on_complete)

    mirror_tap = mrc.MirrorTap("test_mirror_tap")
    pipe = mrc.Pipeline()

    tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main, test_name + "_main_source", test_name + "_main_sink")

    tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored, test_name + "_mirror_sink")

    pipe.make_segment("segment_main", [], [mirror_tap.get_egress_tap_name()], tapped_init_wrapper_main)
    pipe.make_segment("segment_mirror", [mirror_tap.get_ingress_tap_name()], [], tapped_init_wrapper_mirrored)

    options = mrc.Options()

    executor = mrc.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()

    assert (packets_main == packet_count)
    assert (packets_mirrored >= packet_count * 0.5)


def test_single_pipeline_tap_and_buffer_with_additional_ports():
    global packet_count, packets_main, packets_mirrored, packets_non_mirrored
    packet_count, packets_main, packets_mirrored, packets_non_mirrored = 10000, 0, 0, 0

    global test_name
    test_name = "test_single_pipeline_tap_and_buffer_with_additional_ports"

    def gen_data_one():
        global packet_count
        for i in range(packet_count):
            yield i

    def gen_data_two():
        global packet_count
        for i in range(packet_count):
            yield i

    def init_wrapper_main(builder: mrc.Builder):

        def on_next_sink(input):
            global packets_main
            packets_main += 1

        def on_error():
            pass

        def on_complete():
            pass

        source = builder.make_source(test_name + "_main_source", gen_data_one)
        sink = builder.make_sink(test_name + "_main_sink", on_next_sink, on_error, on_complete)
        builder.make_edge(source, sink)

        source = builder.make_source(test_name + "_main_extra_source", gen_data_two)

        egress = builder.get_egress("non_mirror_port")
        builder.make_edge(source, egress)

    def init_wrapper_mirrored(builder: mrc.Builder):

        def on_next_sink(input):
            global packets_mirrored
            packets_mirrored += 1

        def non_mirror_sink(input):
            global packets_non_mirrored
            packets_non_mirrored += 1

        def on_error():
            pass

        def on_complete():
            pass

        builder.make_sink(test_name + "_mirror_sink", on_next_sink, on_error, on_complete)

        ingress = builder.get_ingress("non_mirror_port")
        non_mirror_sink = builder.make_sink(test_name + "_mirror_extra_sink", non_mirror_sink, on_error, on_complete)

        builder.make_edge(ingress, non_mirror_sink)

    mirror_tap = mrc.MirrorTap("test_mirror_tap")
    pipe = mrc.Pipeline()

    tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main, test_name + "_main_source", test_name + "_main_sink")

    tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored, test_name + "_mirror_sink")

    egress_ports = mirror_tap.create_or_extend_egress_ports(["non_mirror_port"])
    pipe.make_segment("segment_main", [], egress_ports, tapped_init_wrapper_main)

    ingress_ports = mirror_tap.create_or_extend_ingress_ports(["non_mirror_port"])
    pipe.make_segment("segment_mirror", ingress_ports, [], tapped_init_wrapper_mirrored)

    options = mrc.Options()

    executor = mrc.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()

    assert (packets_main == packet_count)
    assert (packets_non_mirrored == packet_count)
    assert (packets_mirrored >= packet_count * 0.5)


def test_single_pipeline_tap_and_buffer_with_module():
    global packet_count, packets_main, packets_mirrored
    packet_count, packets_main, packets_mirrored = 10000, 0, 0

    global test_name
    test_name = "test_single_pipeline_tap_and_buffer"

    registry = mrc.ModuleRegistry
    release_version = [int(x) for x in mrc.__version__.split(".")]

    old_release_version = [22, 10, 0]
    no_version_patch = [22, 10]
    no_version_minor_and_patch = [22]

    assert registry.is_version_compatible(release_version)
    assert registry.is_version_compatible(old_release_version) is not True
    assert registry.is_version_compatible(no_version_patch) is not True
    assert registry.is_version_compatible(no_version_minor_and_patch) is not True

    tap_name = "py_test_mirror_module_tap"

    def gen_data_one():
        global packet_count
        for i in range(packet_count):
            yield {"data": i}

    def init_wrapper_main(builder: mrc.Builder):
        mirror_module_id = "MirrorTap"
        mirror_module_ns = "mrc"

        config = {"tap_id_override": tap_name}
        mirror_tap_module = builder.load_module(mirror_module_id, mirror_module_ns, test_name + "_mirror_tap", config)

        def on_next_sink(input):
            global packets_main
            packets_main += 1

        def on_error():
            pass

        def on_complete():
            pass

        source = builder.make_source(test_name + "_main_source", gen_data_one())
        sink = builder.make_sink(test_name + "_main_sink", on_next_sink, on_error, on_complete)

        builder.make_edge(source, mirror_tap_module.input_port("input"))
        builder.make_edge(mirror_tap_module.output_port("output"), sink)

    def init_wrapper_mirrored(builder: mrc.Builder):
        stream_buffer_model_id = "MirrorStreamBufferImmediate"
        stream_buffer_ns = "mrc"

        config = {"buffer_size": 1024, "tap_id_override": tap_name}
        stream_buffer_module = builder.load_module(stream_buffer_model_id,
                                                   stream_buffer_ns,
                                                   "test_mirror_stream",
                                                   config)

        def on_next_sink(input):
            global packets_mirrored
            packets_mirrored += 1

        def on_error():
            pass

        def on_complete():
            pass

        sink = builder.make_sink(test_name + "_mirror_sink", on_next_sink, on_error, on_complete)
        builder.make_edge(stream_buffer_module.output_port("output"), sink)

    pipe = mrc.Pipeline()

    pipe.make_segment("segment_main", [], [tap_name], init_wrapper_main)
    pipe.make_segment("segment_mirror", [tap_name], [], init_wrapper_mirrored)

    options = mrc.Options()

    executor = mrc.Executor(options)
    executor.register_pipeline(pipe)

    executor.start()
    executor.join()

    assert (packets_main == packet_count)
    assert (packets_mirrored >= packet_count * 0.5)


if (__name__ == "__main__"):
    test_mirror_tap_init()
    test_single_pipeline_tap_and_buffer()
    test_single_pipeline_tap_and_buffer_with_additional_ports()
    test_single_pipeline_tap_and_buffer_with_module()
