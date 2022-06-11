import srf


def run_pipeline():

    counter = 0

    def segment_init(seg: srf.Builder):

        # Use a generator function as the source
        def source_gen():

            yield 1
            yield 2
            yield 3

        # Create the source object
        src = seg.make_source("int_source", source_gen())

        # This method will get called each time the sink gets a value
        def sink_on_next(x):

            nonlocal counter
            counter += 1

        # Build the sink object
        sink = seg.make_sink("int_sink", sink_on_next, None, None)

        # Connect the source to the sink
        seg.make_edge(src, sink)

    # Create the pipeline object
    pipeline = srf.Pipeline()

    # Create a segment
    pipeline.make_segment("my_seg", segment_init)

    # Build executor options
    options = srf.Options()

    # Set to 1 thread
    options.topology.user_cpuset = "0-0"

    # Create the executor
    executor = srf.Executor(options)

    # Register pipeline to tell executor what to run
    executor.register_pipeline(pipeline)

    print("srf pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    print("srf pipeline complete: counter should be 3; counter={}".format(counter))


if (__name__ == "__main__"):
    run_pipeline()
