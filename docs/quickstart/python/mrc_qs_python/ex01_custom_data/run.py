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

import dataclasses

import mrc


@dataclasses.dataclass
class MyCustomClass:
    """
    This is our custom data class that is used to store both a name and a value.
    """

    value: int
    name: str


def run_pipeline():

    # This variable will be used to store the sum of all emitted values at the sink
    total_sum = 0

    def segment_init(seg: mrc.Builder):

        # Use a generator function as the source
        def source_gen():

            for i in range(3):

                # Emit our custom object here giving it a name
                yield MyCustomClass(i, "Instance-{}".format(i))

        # Create the source object
        src = seg.make_source("source", source_gen())

        def update_obj(x: MyCustomClass):

            print("Processing '{}'".format(x.name))

            # Alter the value property of the class
            x.value = x.value * 2

            return x

        # Make an intermediate node
        node = seg.make_node("node", update_obj)

        # Connect source to node
        seg.make_edge(src, node)

        # This method will get called each time the sink gets a value
        def sink_on_next(x: MyCustomClass):

            nonlocal total_sum
            total_sum += x.value

        # Build the sink object
        sink = seg.make_sink("sink", sink_on_next, None, None)

        # Connect the source to the sink
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

    print("mrc pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    print("mrc pipeline complete: total_sum should be 6; total_sum={}".format(total_sum))


if (__name__ == "__main__"):
    run_pipeline()
