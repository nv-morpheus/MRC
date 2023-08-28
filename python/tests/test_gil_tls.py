# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import threading

import mrc
from mrc.tests.utils import RequireGilInDestructor

TLS = threading.local()


def test_gc_called_in_thread_finalizer():
    """
    Test to reproduce issue #362
    No asserts needed if it doesn't segfault, then we're good
    """
    mrc.logging.log("Building pipeline")

    def source_gen():
        mrc.logging.log("source_gen")
        x = RequireGilInDestructor()
        TLS.x = x
        yield x

    def init_seg(builder: mrc.Builder):
        builder.make_source("souce_gen", source_gen)

    pipe = mrc.Pipeline()
    pipe.make_segment("seg1", init_seg)

    options = mrc.Options()
    executor = mrc.Executor(options)
    executor.register_pipeline(pipe)
    executor.start()
    executor.join()
