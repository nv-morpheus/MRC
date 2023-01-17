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

import typing

import pytest


@pytest.fixture(scope="session")
def is_debugger_attached():
    import sys

    trace_func = sys.gettrace()

    # The presence of a trace function and pydevd means a debugger is attached
    if (trace_func is not None):
        trace_module = getattr(trace_func, "__module__", None)

        if (trace_module is not None and trace_module.find("pydevd") != -1):
            return True

    return False


@pytest.fixture(scope="session", autouse=True)
def configure_tests_logging(is_debugger_attached: bool):
    """
    Sets the base logging settings for the entire test suite to ensure logs are generated. Automatically detects if a
    debugger is attached and lowers the logging level to DEBUG.
    """
    import logging

    from mrc.core import logging as mrc_logging

    log_level = logging.WARN

    # Check if a debugger is attached. If so, choose DEBUG for the logging level. Otherwise, only WARN
    if (is_debugger_attached):
        log_level = logging.INFO

    mrc_logging.init_logging("mrc_testing", py_level=log_level)


@pytest.fixture
def single_segment_defaults() -> dict:
    return {
        "segment_name": "my_seg",
        "user_cpuset": "0-0",
    }


@pytest.fixture(scope="function")
def single_segment_pipeline(single_segment_defaults: dict):

    import mrc

    segment_config: dict = single_segment_defaults.copy()

    def run_exec(config: dict, segment_init: typing.Callable[[mrc.Builder], None]):

        segment_config.update(config)

        pipeline = mrc.Pipeline()

        pipeline.make_segment(segment_config["segment_name"], segment_init)

        options = mrc.Options()

        # Set to 1 thread
        options.topology.user_cpuset = segment_config["user_cpuset"]

        executor = mrc.Executor(options)

        executor.register_pipeline(pipeline)

        executor.start()

        executor.join()

    return run_exec


# SingleSegmentPipeine = typing.Callable[[dict, typing.Callable[[mrc.Builder], None]], None]
