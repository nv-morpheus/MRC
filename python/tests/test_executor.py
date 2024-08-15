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

import asyncio
import os
import time
import typing

import pytest

import mrc
from mrc.tests.utils import throw_cpp_error


def pairwise(t):
    it = iter(t)
    return zip(it, it)


node_fn_type = typing.Callable[[mrc.Builder], mrc.SegmentObject]


@pytest.fixture
def source():

    def build(builder: mrc.Builder):

        def gen_data():
            yield 1
            yield 2
            yield 3

        return builder.make_source("source", gen_data)

    return build


@pytest.fixture
def endless_source():

    def build(builder: mrc.Builder):

        def gen_data():
            i = 0
            while True:
                yield i
                i += 1
                time.sleep(0.1)

        return builder.make_source("endless_source", gen_data())

    return build


@pytest.fixture
def blocking_source():

    def build(builder: mrc.Builder):

        def gen_data():
            yield 1
            while True:
                time.sleep(0.1)

        return builder.make_source("blocking_source", gen_data)

    return build


@pytest.fixture
def source_pyexception():

    def build(builder: mrc.Builder):

        def gen_data_and_raise():
            yield 1
            yield 2
            yield 3

            raise RuntimeError("Raised python error")

        return builder.make_source("source", gen_data_and_raise)

    return build


@pytest.fixture
def source_cppexception():

    def build(builder: mrc.Builder):

        def gen_data_and_raise():
            yield 1
            yield 2
            yield 3

            throw_cpp_error()

        return builder.make_source("source", gen_data_and_raise)

    return build


@pytest.fixture
def node_exception():

    def build(builder: mrc.Builder):

        def on_next(data):
            print("Received value: {}".format(data), flush=True)
            raise RuntimeError("unittest")

        return builder.make_node("node", mrc.core.operators.map(on_next))

    return build


@pytest.fixture
def sink():

    def build(builder: mrc.Builder):

        def sink_on_next(data):
            print("Got value: {}".format(data), flush=True)

        return builder.make_sink("sink", sink_on_next, None, None)

    return build


@pytest.fixture
def build_pipeline():

    def inner(*node_fns: node_fn_type):

        def init_segment(builder: mrc.Builder):

            created_nodes = []

            # Loop over node creation functions
            for n in node_fns:
                created_nodes.append(n(builder))

            # For each pair, call make_edge
            for source, sink in pairwise(created_nodes):
                builder.make_edge(source, sink)

        pipe = mrc.Pipeline()

        pipe.make_segment("TestSegment11", init_segment)

        return pipe

    return inner


build_pipeline_type = typing.Callable[[typing.Tuple[node_fn_type, ...]], mrc.Pipeline]


@pytest.fixture
def build_executor():

    def inner(pipe: mrc.Pipeline):
        options = mrc.Options()
        options.topology.user_cpuset = f"0-{os.cpu_count() - 1}"
        options.engine_factories.default_engine_type = mrc.core.options.EngineType.Thread
        executor = mrc.Executor(options)
        executor.register_pipeline(pipe)

        executor.start()

        return executor

    return inner


build_executor_type = typing.Callable[[mrc.Pipeline], mrc.Executor]


def test_pyexception_in_source(source_pyexception: node_fn_type,
                               sink: node_fn_type,
                               build_pipeline: build_pipeline_type,
                               build_executor: build_executor_type):

    pipe = build_pipeline(source_pyexception, sink)

    executor = build_executor(pipe)

    with pytest.raises(RuntimeError):
        executor.join()


def test_cppexception_in_source(source_cppexception: node_fn_type,
                                sink: node_fn_type,
                                build_pipeline: build_pipeline_type,
                                build_executor: build_executor_type):

    pipe = build_pipeline(source_cppexception, sink)

    executor = build_executor(pipe)

    with pytest.raises(RuntimeError):
        executor.join()


def test_pyexception_in_source_async(source_pyexception: node_fn_type,
                                     sink: node_fn_type,
                                     build_pipeline: build_pipeline_type,
                                     build_executor: build_executor_type):

    pipe = build_pipeline(source_pyexception, sink)

    async def run_pipeline():
        executor = build_executor(pipe)

        with pytest.raises(RuntimeError):
            await executor.join_async()

    asyncio.run(run_pipeline())


def test_cppexception_in_source_async(source_cppexception: node_fn_type,
                                      sink: node_fn_type,
                                      build_pipeline: build_pipeline_type,
                                      build_executor: build_executor_type):

    pipe = build_pipeline(source_cppexception, sink)

    async def run_pipeline():
        executor = build_executor(pipe)

        with pytest.raises(RuntimeError):
            await executor.join_async()

    asyncio.run(run_pipeline())


@pytest.mark.parametrize("souce_name", ["source", "endless_source", "blocking_source"])
def test_pyexception_in_node(source: node_fn_type,
                             endless_source: node_fn_type,
                             blocking_source: node_fn_type,
                             node_exception: node_fn_type,
                             build_pipeline: build_pipeline_type,
                             build_executor: build_executor_type,
                             souce_name: str):
    """
    Test to reproduce Morpheus issue #1838 where an exception raised in a node doesn't always shutdown the executor
    when the source is intended to run indefinitely.
    """

    if souce_name == "endless_source":
        source_fn = endless_source
    elif souce_name == "blocking_source":
        source_fn = blocking_source
    else:
        source_fn = source

    pipe = build_pipeline(source_fn, node_exception)

    executor: mrc.Executor = None

    executor = build_executor(pipe)

    with pytest.raises(RuntimeError):
        executor.join()


if (__name__ in ("__main__", )):
    test_pyexception_in_source()
