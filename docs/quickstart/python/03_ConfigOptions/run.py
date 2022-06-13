import argparse
import threading

import srf


def run_pipeline(count: int, channel_size: int, threads: int):
    def segment_init(seg: srf.Builder):

        # Use a generator function as the source
        def source_gen():

            print("Source: Starting")
            for i in range(count):

                yield i
                print("Source: Emitted    {:02d}, TID: [{}]".format(i, threading.current_thread().getName()))

            print("Source: Complete")

        # Create the source object
        src = seg.make_source("int_source", source_gen())

        def update_obj(x: int):

            print("Node  : Processing {:02d}, TID: [{}]".format(x, threading.current_thread().getName()))
            return x

        # Make an intermediate node
        node = seg.make_node("node", update_obj)

        # Connect source to node
        seg.make_edge(src, node)

        # This method will get called each time the sink gets a value
        def sink_on_next(x: int):

            print("Sink  : Got value  {:02d}, TID: [{}]".format(x, threading.current_thread().getName()))

        # Build the sink object
        sink = seg.make_sink("int_sink", sink_on_next, None, None)

        # Connect the source to the sink
        seg.make_edge(node, sink)

    srf.Config.default_channel_size = channel_size

    # Create the pipeline object
    pipeline = srf.Pipeline()

    # Create a segment
    pipeline.make_segment("my_seg", segment_init)

    # Build executor options
    options = srf.Options()

    # Set the number of cores to use. Uses the format `{min_core}-{max_core}` (inclusive)
    options.topology.user_cpuset = "0-{}".format(threads - 1)

    # Create the executor
    executor = srf.Executor(options)

    # Register pipeline to tell executor what to run
    executor.register_pipeline(pipeline)

    print("srf pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    print("srf pipeline complete.".format())


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(description='ConfigOptions Example.')
    parser.add_argument('--count', type=int, default=10, help="The number of items for the source to emit")
    parser.add_argument('--channel_size',
                        type=int,
                        default=4,
                        help="The size of the inter-node buffers. Must be a power of 2")
    parser.add_argument('--threads', type=int, default=1, help="The number of threads to use.")

    args = parser.parse_args()

    run_pipeline(args.count, args.channel_size, args.threads)
