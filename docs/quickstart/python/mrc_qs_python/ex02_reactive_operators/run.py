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
from mrc.core import operators as ops


@dataclasses.dataclass
class MyCustomClass:

    value: int
    name: str


def run_pipeline():

    def segment_init(seg: mrc.Builder):

        # Use a generator function as the source
        def source_gen():

            for i in range(5):

                yield MyCustomClass(i, "Instance-{}".format(i))

        # Create the source object
        src = seg.make_source("source", source_gen())

        value_count = 0
        value_sum = 0

        def node_fn(input: mrc.Observable, output: mrc.Subscriber):

            def update_obj(x: MyCustomClass):
                nonlocal value_count
                nonlocal value_sum

                # Alter the value property of the class
                x.value = x.value * 2

                # Update the sum values
                value_count += 1
                value_sum += x.value

                return x

            def on_completed():

                # Prevent divide by 0. Just in case
                if (value_count <= 0):
                    return

                return MyCustomClass(value_sum / value_count, "Mean")

            input.pipe(ops.filter(lambda x: x.value % 2 == 0), ops.map(update_obj),
                       ops.on_completed(on_completed)).subscribe(output)

        # Make an intermediate node
        node = seg.make_node_full("node", node_fn)

        # Connect source to node
        seg.make_edge(src, node)

        # This method will get called each time the sink gets a value
        def sink_on_next(x: MyCustomClass):

            print("Sink: Got Obj Name: {}, Value: {}".format(x.name, x.value))

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

    print("mrc pipeline completed.")


if (__name__ == "__main__"):
    run_pipeline()
