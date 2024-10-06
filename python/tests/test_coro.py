# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from mrc._pymrc.tests.coro.coro import call_async
from mrc._pymrc.tests.coro.coro import call_fib_async
from mrc._pymrc.tests.coro.coro import raise_at_depth_async
from mrc.core import coro


@pytest.mark.asyncio
async def test_coro():

    # hit_inside = False

    async def inner():

        # nonlocal hit_inside

        result = await coro.wrap_coroutine(asyncio.sleep(1, result=['a', 'b', 'c']))

        # hit_inside = True

        return [result]

    returned_val = await coro.wrap_coroutine(inner())

    assert returned_val == 'a'
    # assert hit_inside


@pytest.mark.asyncio
async def test_coro_many():

    expected_count = 1000
    hit_count = 0

    start_time = asyncio.get_running_loop().time()

    async def inner():

        nonlocal hit_count

        await asyncio.sleep(0.1)

        hit_count += 1

        return ['a', 'b', 'c']

    coros = [coro.wrap_coroutine(inner()) for _ in range(expected_count)]

    returned_vals = await asyncio.gather(*coros)

    end_time = asyncio.get_running_loop().time()

    assert returned_vals == ['a'] * expected_count
    assert hit_count == expected_count
    assert (end_time - start_time) < 1.5


@pytest.mark.asyncio
async def test_python_cpp_async_interleave():

    def fib(n):
        if n < 0:
            raise ValueError()

        if n < 2:
            return 1

        return fib(n - 1) + fib(n - 2)

    async def fib_async(n):
        if n < 0:
            raise ValueError()

        if n < 2:
            return 1

        task_a = call_fib_async(fib_async, n, 1)
        task_b = call_fib_async(fib_async, n, 2)

        [a, b] = await asyncio.gather(task_a, task_b)

        return a + b

    assert fib(15) == await fib_async(15)


@pytest.mark.asyncio
async def test_python_cpp_async_exception():

    async def py_raise_at_depth_async(n: int):
        if n <= 0:
            raise RuntimeError("depth reached zero in python")

        await raise_at_depth_async(py_raise_at_depth_async, n - 1)

    depth = 100

    with pytest.raises(RuntimeError) as ex:
        await raise_at_depth_async(py_raise_at_depth_async, depth + 1)
    assert "python" in str(ex.value)

    with pytest.raises(RuntimeError) as ex:
        await raise_at_depth_async(py_raise_at_depth_async, depth)
    assert "c++" in str(ex.value)


@pytest.mark.asyncio
async def test_can_cancel_coroutine_from_python():

    counter = 0

    async def increment_recursively():
        nonlocal counter
        await asyncio.sleep(0)
        counter += 1
        await call_async(increment_recursively)

    task = asyncio.ensure_future(call_async(increment_recursively))

    await asyncio.sleep(0)
    assert counter == 0
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert counter == 1
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert counter == 2

    task.cancel()

    with pytest.raises(asyncio.exceptions.CancelledError):
        await task

    assert counter == 3
