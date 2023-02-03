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

import mrc

VERSION = [int(cmpt) for cmpt in mrc.__version__.split(".")]


def test_mirror_tap_init():
    mirror_tap_one = mrc.MirrorTap("test_mirror_tap")
    mirror_tap_two = mrc.MirrorTap("test_mirror_tap", {"test": "test"})


def test_single_pipeline_tap_and_buffer():
    global packet_count, packets_main, packets_mirrored
    packet_count, packets_main, packets_mirrored = 10000, 0, 0

    global test_name
    test_name = "test_single_pipeline_tap_and_buffer"

    def gen_data():
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

        source = builder.make_source(test_name + "_main_source", gen_data())
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

    tapped_init_wrapper_main = mirror_tap.tap(init_wrapper_main,
                                              test_name + "_main_source",
                                              test_name + "_main_sink")

    tapped_init_wrapper_mirrored = mirror_tap.stream_to(init_wrapper_mirrored,
                                                        test_name + "_mirror_sink")

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
    pass