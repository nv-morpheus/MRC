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

import functools

import pytest

import mrc
import mrc.core.operators as ops


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


@pytest.mark.parametrize("engines_per_pe", [1, 2])
@pytest.mark.parametrize("pe_count", [1, 3])
@pytest.mark.parametrize("source_type", ["iterator", "iterable", "function"])
def test_launch_options_source(source_type: str, pe_count: int, engines_per_pe: int):
    hit_count = 0

    source = None

    def source_gen():
        yield int(1)
        yield int(2)
        yield int(3)

    if (source_type == "iterator"):
        if (pe_count > 1 or engines_per_pe > 1):
            # Currently, errors that occur in pipeline threads do not bubble back up to python and simply cause a
            # segfault. Multi-threaded iterator sources intentionally throw an exception that should be tested. However,
            # there is no current way to catch that error on the python side and test for it. Until that is fixed, these
            # tests will simply be skipped
            pytest.skip("Skipping multi-thread iterator sources until pipeline errors can be caught in python.")

        # Create a single instance of the generator
        source = source_gen()
    elif (source_type == "iterable"):
        # Use an iterable object
        source = [1, 2, 3]
    elif (source_type == "function"):
        # Use a factory function to make the generator
        source = source_gen

    def segment_init(seg: mrc.Builder):

        src_node = seg.make_source("my_src", source)

        src_node.launch_options.pe_count = pe_count
        src_node.launch_options.engines_per_pe = engines_per_pe

        def node_fn(x: int):
            nonlocal hit_count

            hit_count += 1

        hit_counter = seg.make_node("hit_counter", node_fn)
        seg.make_edge(src_node, hit_counter)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-{}".format(pe_count)

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    if (source_type == "iterator"):
        # Cant restart iterators. So only 1 loop is expected
        pe_count = 1
        engines_per_pe = 1

    assert hit_count == 3 * pe_count * engines_per_pe


@pytest.mark.parametrize("engines_per_pe", [1])
@pytest.mark.parametrize("pe_count", [1])
@pytest.mark.parametrize("source_type", ["iterator", "iterable", "function"])
def test_launch_options_source_component(source_type: str, pe_count: int, engines_per_pe: int):
    hit_count = 0

    source = None

    def source_gen():
        yield int(1)
        yield int(2)
        yield int(3)

    if (source_type == "iterator"):
        if (pe_count > 1 or engines_per_pe > 1):
            # Currently, errors that occur in pipeline threads do not bubble back up to python and simply cause a
            # segfault. Multi-threaded iterator sources intentionally throw an exception that should be tested. However,
            # there is no current way to catch that error on the python side and test for it. Until that is fixed, these
            # tests will simply be skipped
            pytest.skip("Skipping multi-thread iterator sources until pipeline errors can be caught in python.")

        # Create a single instance of the generator
        source = source_gen()
    elif (source_type == "iterable"):
        # Use an iterable object
        source = [1, 2, 3]
    elif (source_type == "function"):
        # Use a factory function to make the generator
        source = source_gen

    def segment_init(seg: mrc.Builder):

        src_node = seg.make_source_component("my_src", source)

        # src_node.launch_options.pe_count = pe_count
        # src_node.launch_options.engines_per_pe = engines_per_pe

        def node_fn(x: int):
            nonlocal hit_count

            hit_count += 1

        hit_counter = seg.make_node("hit_counter", node_fn)
        seg.make_edge(src_node, hit_counter)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-{}".format(pe_count)

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    if (source_type == "iterator"):
        # Cant restart iterators. So only 1 loop is expected
        pe_count = 1
        engines_per_pe = 1

    assert hit_count == 3 * pe_count * engines_per_pe


def test_launch_options_iterable():
    pe_count = 2
    engines_per_pe = 4

    hit_count = 0

    def segment_init(seg: mrc.Builder):
        src_node = seg.make_source("my_src", [1, 2, 3])

        src_node.launch_options.pe_count = pe_count
        src_node.launch_options.engines_per_pe = engines_per_pe

        def node_fn(x: int):
            nonlocal hit_count

            hit_count += 1

        hit_counter = seg.make_node("hit_counter", node_fn)
        seg.make_edge(src_node, hit_counter)

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-{}".format(pe_count)

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()

    assert hit_count == 3 * pe_count * engines_per_pe


# @pytest.mark.skip(reason="#172 - awaiting multi-pe safe source work around")
def test_launch_options_properties():

    def segment_init(seg: mrc.Builder):

        def source_gen():
            yield int(1)

        src_node = seg.make_source("my_src", source_gen)

        # Create a simple sink to avoid errors
        sink_node = seg.make_sink("my_sink", lambda x: None, None, None)

        seg.make_edge(src_node, sink_node)

        assert src_node.launch_options.pe_count == 1, "Default should be 1"
        assert src_node.launch_options.engines_per_pe == 1, "Default should be 1"
        assert src_node.launch_options.engine_factory_name == "default", "Default should be 'default'"

        src_node.launch_options.pe_count = 2
        assert src_node.launch_options.pe_count == 2, "Set and get should match"

        src_node.launch_options.engines_per_pe = 2
        assert src_node.launch_options.engines_per_pe == 2, "Set and get should match"

        src_node.launch_options.engine_factory_name = "fiber"
        assert src_node.launch_options.engine_factory_name == "fiber", "Set and get should match"

        # Save a reference
        lo = src_node.launch_options

        src_node.launch_options.pe_count = 5
        assert lo.pe_count == 5, "Reference launch_options should match"

        lo.pe_count = 3
        assert src_node.launch_options.pe_count == 3, "Should be able to set from referenced object"

        src_node.launch_options.engines_per_pe = 5
        assert lo.engines_per_pe == 5, "Reference launch_options should match"

        lo.engines_per_pe = 3
        assert src_node.launch_options.engines_per_pe == 3, "Should be able to set from referenced object"

        src_node.launch_options.engine_factory_name = "thread"
        assert lo.engine_factory_name == "thread", "Reference launch_options should match"

        lo.engine_factory_name = "fiber"
        assert src_node.launch_options.engine_factory_name == "fiber", "Should be able to set from referenced object"

        # Reset it back to default
        src_node.launch_options.engine_factory_name = "default"

    pipeline = mrc.Pipeline()

    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-5"

    executor = mrc.Executor(options)

    executor.register_pipeline(pipeline)

    executor.start()

    executor.join()


@pytest.mark.parametrize("use_on_completed", [True, False])
@pytest.mark.parametrize("use_on_error", [True, False])
@pytest.mark.parametrize("use_on_next", [True, False])
@pytest.mark.parametrize("node_type", ["runnable", "component"])
def test_sink_function_options(single_segment_pipeline,
                               node_type: str,
                               use_on_next: bool,
                               use_on_error: bool,
                               use_on_completed: bool):

    is_component = node_type == "component"

    on_next_count = 0
    on_error_count = 0
    on_completed_count = 0

    def segment_init(seg: mrc.Builder):

        def create_source():
            for i in range(5):
                yield i

        source = seg.make_source("source", create_source())

        def on_next(x: int):
            nonlocal on_next_count
            on_next_count += 1

        def on_error(e):
            nonlocal on_error_count
            on_error_count += 1

        def on_completed():
            nonlocal on_completed_count
            on_completed_count += 1

        on_next_fn = on_next if use_on_next else None
        on_error_fn = on_error if use_on_error else None
        on_completed_fn = on_completed if use_on_completed else None

        if (is_component):
            sink = seg.make_sink_component("sink", on_next_fn, on_error_fn, on_completed_fn)
        else:
            sink = seg.make_sink("sink", on_next_fn, on_error_fn, on_completed_fn)

        seg.make_edge(source, sink)

    single_segment_pipeline({"user_cpuset": "0-1"}, segment_init)

    assert on_next_count == (5 if use_on_next else 0)
    assert on_error_count == (0 if use_on_error else 0)
    assert on_completed_count == (1 if use_on_completed else 0)


@pytest.mark.parametrize("use_partial", [True, False])
@pytest.mark.parametrize("node_type", ["runnable", "component"])
def test_node_function_unpacking(single_segment_pipeline, node_type: str, use_partial: bool):

    is_component = node_type == "component"

    on_next_values = {}
    on_error_count = 0
    on_completed_count = 0

    max_emissions = 5

    def segment_init(seg: mrc.Builder):

        def create_source():
            for s_idx, s in enumerate(["one", "two", "three"]):
                for i in range(s_idx, 5):
                    if (use_partial):
                        yield s, i + s_idx
                    else:
                        yield s, i + s_idx, max_emissions

        source = seg.make_source("source", create_source())

        def node_on_next(s: str, i: int, max_e: int):

            # Double both values
            return s + s, i + i, max_e

        if (use_partial):
            node_on_next = functools.partial(node_on_next, max_e=max_emissions)

        if (is_component):
            node = seg.make_node_component("node", ops.map(node_on_next))
        else:
            node = seg.make_node("node", ops.map(node_on_next))

        seg.make_edge(source, node)

        def on_next(s: str, i: int, max_e: int):
            nonlocal on_next_values

            if (s not in on_next_values):
                on_next_values[s] = []

            on_next_values[s].append(i)

            # Make sure this value is set
            assert max_emissions == max_e

        def on_error(e):
            nonlocal on_error_count
            on_error_count += 1

        def on_completed():
            nonlocal on_completed_count
            on_completed_count += 1

        if (is_component):
            sink = seg.make_sink_component("sink", on_next, on_error, on_completed)
        else:
            sink = seg.make_sink("sink", on_next, on_error, on_completed)

        seg.make_edge(node, sink)

    single_segment_pipeline({"user_cpuset": "0-1"}, segment_init)

    assert on_next_values == {
        "oneone": [0, 2, 4, 6, 8],
        "twotwo": [4, 6, 8, 10],
        "threethree": [8, 10, 12],
    }
    assert on_error_count == 0
    assert on_completed_count == 1


@pytest.mark.parametrize("use_partial", [True, False])
@pytest.mark.parametrize("node_type", ["runnable", "component"])
def test_sink_function_unpacking(single_segment_pipeline, node_type: str, use_partial: bool):

    is_component = node_type == "component"

    on_next_values = {}
    on_error_count = 0
    on_completed_count = 0

    max_emissions = 5

    def segment_init(seg: mrc.Builder):

        def create_source():
            for s_idx, s in enumerate(["one", "two", "three"]):
                for i in range(s_idx, max_emissions):
                    if (use_partial):
                        yield s, i + s_idx
                    else:
                        yield s, i + s_idx, max_emissions

        source = seg.make_source("source", create_source())

        def on_next(s: str, i: int, max_e: int):
            nonlocal on_next_values

            if (s not in on_next_values):
                on_next_values[s] = []

            on_next_values[s].append(i)

            # Make sure this value is set
            assert max_emissions == max_e

        def on_error(e):
            nonlocal on_error_count
            on_error_count += 1

        def on_completed():
            nonlocal on_completed_count
            on_completed_count += 1

        if (use_partial):
            on_next = functools.partial(on_next, max_e=max_emissions)

        if (is_component):
            sink = seg.make_sink_component("sink", on_next, on_error, on_completed)
        else:
            sink = seg.make_sink("sink", on_next, on_error, on_completed)

        seg.make_edge(source, sink)

    single_segment_pipeline({"user_cpuset": "0-1"}, segment_init)

    assert on_next_values == {
        "one": [0, 1, 2, 3, 4],
        "two": [2, 3, 4, 5],
        "three": [4, 5, 6],
    }
    assert on_error_count == 0
    assert on_completed_count == 1


def test_invalid_source():

    def segment_init(seg: mrc.Builder):

        def source_gen(a, b):
            yield a + b

        seg.make_source("my_src", source_gen)

    pipeline = mrc.Pipeline()
    pipeline.make_segment("my_seg", segment_init)

    options = mrc.Options()
    executor = mrc.Executor(options)
    executor.register_pipeline(pipeline)

    with pytest.raises(ValueError):
        executor.start()
        executor.join()


if (__name__ == "__main__"):
    test_launch_options_properties()
