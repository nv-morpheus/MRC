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

import asyncio
import typing

import pytest

import srf
from srf.tests.utils import throw_cpp_error


def pairwise(t):
    it = iter(t)
    return zip(it, it)


node_fn_type = typing.Callable[[srf.Builder], srf.SegmentObject]


@pytest.fixture
def source_pyexception():

    def build(builder: srf.Builder):

        def gen_data_and_raise():
            yield 1
            yield 2
            yield 3

            raise RuntimeError("Raised python error")

        return builder.make_source("source", gen_data_and_raise)

    return build


@pytest.fixture
def source_cppexception():

    def build(builder: srf.Builder):

        def gen_data_and_raise():
            yield 1
            yield 2
            yield 3

            throw_cpp_error()

        return builder.make_source("source", gen_data_and_raise)

    return build


@pytest.fixture
def sink():

    def build(builder: srf.Builder):

        def sink_on_next(data):
            print("Got value: {}".format(data))

        return builder.make_sink("sink", sink_on_next, None, None)

    return build


@pytest.fixture
def build_pipeline():

    def inner(*node_fns: node_fn_type):

        def init_segment(builder: srf.Builder):

            created_nodes = []

            # Loop over node creation functions
            for n in node_fns:
                created_nodes.append(n(builder))

            # For each pair, call make_edge
            for source, sink in pairwise(created_nodes):
                builder.make_edge(source, sink)

        pipe = srf.Pipeline()

        pipe.make_segment("TestSegment11", init_segment)

        return pipe

    return inner


build_pipeline_type = typing.Callable[[typing.Tuple[node_fn_type, ...]], srf.Pipeline]


@pytest.fixture
def build_executor():

    def inner(pipe: srf.Pipeline):
        options = srf.Options()

        executor = srf.Executor(options)
        executor.register_pipeline(pipe)

        executor.start()

        return executor

    return inner


build_executor_type = typing.Callable[[srf.Pipeline], srf.Executor]


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


if (__name__ in ("__main__", )):
    test_pyexception_in_source()
