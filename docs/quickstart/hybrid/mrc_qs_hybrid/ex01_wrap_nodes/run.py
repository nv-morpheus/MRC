# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from mrc_qs_hybrid.common import setup_logger
from mrc_qs_hybrid.ex01_wrap_nodes import MyDataObjectNode
from mrc_qs_hybrid.ex01_wrap_nodes import MyDataObjectSink
from mrc_qs_hybrid.ex01_wrap_nodes import MyDataObjectSource

import mrc

# Setup logging
logger = setup_logger(__file__)


def run_pipeline():

    def segment_init(seg: mrc.Builder):

        # Create the source object
        src = MyDataObjectSource(seg, "src", count=3)

        node = MyDataObjectNode(seg, "node")

        seg.make_edge(src, node)

        sink = MyDataObjectSink(seg, "sink")

        # Connect the source to the sink. You can also connect nodes by name
        seg.make_edge(node, sink)

    # Create the pipeline object
    pipeline = mrc.Pipeline()

    # Create a segment
    pipeline.make_segment("my_seg", segment_init)

    # Build executor options
    options = mrc.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    # Create the executor
    executor = mrc.Executor(options)

    # Register pipeline to tell executor what to run
    executor.register_pipeline(pipeline)

    logger.info("mrc pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    logger.info("mrc pipeline complete")


if (__name__ == "__main__"):
    run_pipeline()
