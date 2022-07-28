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

import pathlib
import random

import srf
import srf.benchmarking
from srf.core.options import PlacementStrategy

import pytest

whereami = pathlib.Path(__file__).parent.resolve()

# Add a little jitter to testing
TEST_ITERATIONS = random.randint(10, 250)


@pytest.fixture(scope="function", autouse=True)
def reset_tracing_stats():
    # Reset the tracing stats before and after each test
    srf.benchmarking.reset_tracing_stats()
    yield
    srf.benchmarking.reset_tracing_stats()


def on_next(x):
    return x


def on_error(x):
    return


def on_completed():
    return


def double_source():
    global TEST_ITERATIONS
    for d in range(TEST_ITERATIONS):
        yield d


def double_float_type1(x):
    return 2 * x


def double_float_type2(x):
    temp = 2 * x

    return temp


def init_double_segment(builder: srf.Builder):
    ## CXX double source with heterogesrfus segment node composition
    python_source_double = builder.make_source("python_source_double", double_source)
    python_node_2x_1 = builder.make_node("python_node_2x_1", double_float_type1)
    builder.make_edge(python_source_double, python_node_2x_1)

    python_node_2x_2 = builder.make_node("python_node_2x_2", double_float_type2)
    builder.make_edge(python_node_2x_1, python_node_2x_2)

    python_node_2x_3 = builder.make_node("python_node_2x_3", double_float_type1)
    builder.make_edge(python_node_2x_2, python_node_2x_3)

    python_sink_double = builder.make_sink("python_sink_double", on_next, on_error, on_completed)
    builder.make_edge(python_node_2x_3, python_sink_double)


def do_stat_gather_test(name, init_function):
    pipeline = srf.Pipeline()
    pipeline.make_segment(name, init_function)

    options = srf.Options()
    options.placement.cpu_strategy = PlacementStrategy.PerMachine

    executor = srf.Executor(options)
    executor.register_pipeline(pipeline)

    executor.start()
    executor.join()


def test_stat_gather_operators():
    srf.benchmarking.reset_tracing_stats()
    srf.benchmarking.trace_operators(True)

    do_stat_gather_test("stat_gather_operators", init_double_segment)
    required_components = [("python_source_double", "src"), ("python_node_2x_1", "internal"),
                           ("python_node_2x_2", "internal"), ("python_node_2x_3", "internal"),
                           ("python_sink_double", "sink")]

    framework_stats_info = srf.benchmarking.get_tracing_stats()
    component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]

    for key, _type in required_components:
        component = component_metrics[key]
        assert (len(component.keys()) > 0)

        if (_type in ("src", "internal")):
            assert (component["component_channel_write_total"] == 0)
            assert (component["component_emissions_total"] == TEST_ITERATIONS)
        if (_type in ("sink", "internal")):
            assert (component["component_channel_read_total"] == 0)
            assert (component["component_receive_total"] == TEST_ITERATIONS)

    srf.benchmarking.reset_tracing_stats()


def test_stat_gather_channels():
    srf.benchmarking.reset_tracing_stats()
    srf.benchmarking.trace_channels(True)

    do_stat_gather_test("stat_gather_channels", init_double_segment)
    required_components = [("python_source_double", "src"), ("python_node_2x_1", "internal"),
                           ("python_node_2x_2", "internal"), ("python_node_2x_3", "internal"),
                           ("python_sink_double", "sink")]

    framework_stats_info = srf.benchmarking.get_tracing_stats()
    component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]

    for key, _type in required_components:
        component = component_metrics[key]
        assert (len(component.keys()) > 0)

        if (_type in ("src", "internal")):
            assert (component["component_channel_write_total"] == TEST_ITERATIONS)
            assert (component["component_emissions_total"] == 0)
        if (_type in ("sink", "internal")):
            assert (component["component_channel_read_total"] == TEST_ITERATIONS)
            assert (component["component_receive_total"] == 0)

    srf.benchmarking.reset_tracing_stats()


def test_stat_gather_full():
    srf.benchmarking.reset_tracing_stats()
    srf.benchmarking.trace_channels(True)
    srf.benchmarking.trace_operators(True)
    do_stat_gather_test("stat_gather_full", init_double_segment)
    required_components = [("python_source_double", "src"), ("python_node_2x_1", "internal"),
                           ("python_node_2x_2", "internal"), ("python_node_2x_3", "internal"),
                           ("python_sink_double", "sink")]

    framework_stats_info = srf.benchmarking.get_tracing_stats()
    component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]

    import json
    print(json.dumps(component_metrics, indent=2))
    for key, _type in required_components:
        component = component_metrics[key]
        assert (len(component.keys()) > 0)
        if (_type in ("src", "internal")):
            assert (component["component_channel_write_total"] == TEST_ITERATIONS)
            assert (component["component_emissions_total"] == TEST_ITERATIONS)
        if (_type in ("sink", "internal")):
            assert (component["component_channel_read_total"] == TEST_ITERATIONS)
            assert (component["component_receive_total"] == TEST_ITERATIONS)

    srf.benchmarking.reset_tracing_stats()


def test_stat_gather_full_noreset():
    srf.benchmarking.reset_tracing_stats()
    srf.benchmarking.trace_channels(True)
    srf.benchmarking.trace_operators(True)
    required_components = [("python_source_double", "src"), ("python_node_2x_1", "internal"),
                           ("python_node_2x_2", "internal"), ("python_node_2x_3", "internal"),
                           ("python_sink_double", "sink")]

    # TODO(devin): If we have two segments in the same binary, with the same node names, they will have their stats
    #   merged. Is this what we want?
    for i in range(1, 5):
        do_stat_gather_test("stat_gather_full_noreset", init_double_segment)

        framework_stats_info = srf.benchmarking.get_tracing_stats()
        component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]

        for key, _type in required_components:
            component = component_metrics[key]

            if (_type in ("src", "internal")):
                assert (component["component_channel_write_total"] == i * TEST_ITERATIONS)
                assert (component["component_emissions_total"] == i * TEST_ITERATIONS)
            if (_type in ("sink", "internal")):
                assert (component["component_channel_read_total"] == i * TEST_ITERATIONS)
                assert (component["component_receive_total"] == i * TEST_ITERATIONS)

    srf.benchmarking.reset_tracing_stats()


def test_stat_gather_full_noreset_start_stop():
    srf.benchmarking.reset_tracing_stats()
    srf.benchmarking.trace_channels(True)
    srf.benchmarking.trace_operators(True)
    required_components = [("python_source_double", "src"), ("python_node_2x_1", "internal"),
                           ("python_node_2x_2", "internal"), ("python_node_2x_3", "internal"),
                           ("python_sink_double", "sink")]

    active_trace_count = 0
    for i in range(1, 10):
        # Randomly pause tracing between runs to verify counts are correct
        pause = random.choice([True, False])
        if (pause):
            srf.benchmarking.trace_operators(False)
            srf.benchmarking.trace_channels(False)
        else:
            active_trace_count += 1

        srf.benchmarking.sync_tracing_state()
        do_stat_gather_test("stat_gather_full_noreset_start_stop", init_double_segment)

        framework_stats_info = srf.benchmarking.get_tracing_stats()
        component_metrics = framework_stats_info["aggregations"]["components"]["metrics"]
        for key, _type in required_components:
            component = component_metrics[key]

            if (_type in ("src", "internal")):
                assert (component["component_channel_write_total"] == active_trace_count * TEST_ITERATIONS)
                assert (component["component_emissions_total"] == active_trace_count * TEST_ITERATIONS)
            if (_type in ("sink", "internal")):
                assert (component["component_channel_read_total"] == active_trace_count * TEST_ITERATIONS)
                assert (component["component_receive_total"] == active_trace_count * TEST_ITERATIONS)

        srf.benchmarking.trace_channels(True)
        srf.benchmarking.trace_operators(True)

    srf.benchmarking.reset_tracing_stats()


if (__name__ in ("__main__", )):
    test_stat_gather_operators()
    test_stat_gather_channels()
    test_stat_gather_full()
    test_stat_gather_full_noreset()
    test_stat_gather_full_noreset_start_stop()
