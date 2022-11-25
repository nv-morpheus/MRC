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
import os
import threading

import mrc


def run_pipeline(count: int, channel_size: int, threads: int):

    def segment_init(seg: mrc.Builder):

        # Use a generator function as the source
        def source_gen():

            print("Source: Starting")
            for i in range(count):

                yield i
                print("Source: Emitted    {:02d}, TID: [{}]".format(i, threading.current_thread().name))

            print("Source: Complete")

        # Create the source object
        src = seg.make_source("int_source", source_gen())

        def update_obj(x: int):

            print("Node  : Processing {:02d}, TID: [{}]".format(x, threading.current_thread().name))
            return x

        # Make an intermediate node
        node = seg.make_node("node", update_obj)

        # Connect source to node
        seg.make_edge(src, node)

        # This method will get called each time the sink gets a value
        def sink_on_next(x: int):

            print("Sink  : Got value  {:02d}, TID: [{}]".format(x, threading.current_thread().name))

        # Build the sink object
        sink = seg.make_sink("int_sink", sink_on_next, None, None)

        # Connect the source to the sink
        seg.make_edge(node, sink)

    mrc.Config.default_channel_size = channel_size

    # Create the pipeline object
    pipeline = mrc.Pipeline()

    # Create a segment
    pipeline.make_segment("my_seg", segment_init)

    # Build executor options
    options = mrc.Options()

    # Set the number of cores to use. Uses the format `{min_core}-{max_core}` (inclusive)
    options.topology.user_cpuset = "0-{}".format(threads - 1)

    # Create the executor
    executor = mrc.Executor(options)

    # Register pipeline to tell executor what to run
    executor.register_pipeline(pipeline)

    print("mrc pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    print("mrc pipeline complete.".format())


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='ConfigOptions Example.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--count', type=int, default=10, help="The number of items for the source to emit")
    parser.add_argument('--channel_size',
                        type=int,
                        default=mrc.Config.default_channel_size,
                        help="The size of the inter-node buffers. Must be a power of 2")
    parser.add_argument('--threads', type=int, default=os.cpu_count(), help="The number of threads to use.")

    args = parser.parse_args()

    run_pipeline(args.count, args.channel_size, args.threads)
