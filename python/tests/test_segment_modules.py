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

        simple_mod = builder.make_module("ModuleEndToEndTest_mod1", "SimpleModule", {})
        configurable_mod = builder.make_module("ModuleEndToEndTest_mod2", "ConfigurableModule", {})

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


if (__name__ in ("__main__",)):
    test_py_end_to_end()
