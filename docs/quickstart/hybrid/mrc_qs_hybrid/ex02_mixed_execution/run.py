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

import argparse

from mrc_qs_hybrid.common import DataObjectNode
from mrc_qs_hybrid.common import DataObjectSink
from mrc_qs_hybrid.common import DataObjectSource
from mrc_qs_hybrid.common import setup_logger
from mrc_qs_hybrid.common.data import DataObject

import mrc

# Setup logging
logger = setup_logger(__file__)


def run_pipeline(python_source: bool, python_node: bool, python_sink: bool):

    def segment_init(seg: mrc.Builder):

        # Create the source object
        # Use a generator function as the source
        def source_gen():

            for i in range(3):

                # Emit our custom object here giving it a name
                yield DataObject("[Python]Instance-{}".format(i), i)

        # Create the source depending on the runtime option
        if (python_source):
            src = seg.make_source("source", source_gen())
        else:
            src = DataObjectSource(seg, "src", count=3)

        def update_obj(x: DataObject):

            logger.info("[Python] Processing '{}'".format(x.name))

            # Alter the value property of the class
            x.value = x.value * 2

            return x

        # Create the node depending on the runtime option
        if (python_node):
            node = seg.make_node("node", update_obj)
        else:
            node = DataObjectNode(seg, "node")

        seg.make_edge(src, node)

        def sink_on_next(x: DataObject):

            # nonlocal value is needed since we are modifying a value outside of our scope
            logger.info("[Python] Got value: {}".format(x))

        # Create the sink depending on the runtime option
        if (python_sink):
            sink = seg.make_sink("sink", sink_on_next, None, None)
        else:
            sink = DataObjectSink(seg, "sink")

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
    parser = argparse.ArgumentParser(description='mixed_execution Example.')
    parser.add_argument('--python_source',
                        action='store_const',
                        const=True,
                        help="Specifying this argument will run the pipeline with a python source")
    parser.add_argument('--python_node',
                        action='store_const',
                        const=True,
                        help="Specifying this argument will run the pipeline with a python node")
    parser.add_argument('--python_sink',
                        action='store_const',
                        const=True,
                        help="Specifying this argument will run the pipeline with a python sink")

    args = parser.parse_args()

    run_pipeline(args.python_source, args.python_node, args.python_sink)
