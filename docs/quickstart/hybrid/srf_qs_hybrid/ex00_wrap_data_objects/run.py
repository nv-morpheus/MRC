import srf
from srf_qs_hybrid.common import setup_logger
from srf_qs_hybrid.ex00_wrap_data_objects import MyDataObject

# Setup logging
logger = setup_logger(__file__)


def run_pipeline():

    # This variable will be used to store the sum of all emitted values at the sink
    total_sum = 0

    def segment_init(seg: srf.Builder):

        # Use a generator function as the source
        def source_gen():

            for i in range(3):

                # Emit our custom object here giving it a name
                yield MyDataObject("Instance-{}".format(i), i)

        # Create the source object
        src = seg.make_source("source", source_gen())

        def update_obj(x: MyDataObject):

            logger.info("Processing '{}'".format(x.name))

            # Alter the value property of the class
            x.value = x.value * 2

            return x

        # Make an intermediate node
        node = seg.make_node("node", update_obj)

        seg.make_edge(src, node)

        # This method will get called each time the sink gets a value
        def sink_on_next(x: MyDataObject):

            # nonlocal value is needed since we are modifying a value outside of our scope
            logger.info("Got value: {}, Incrementing counter".format(x))

            nonlocal total_sum
            total_sum += x.value

        # Build the sink object
        sink = seg.make_sink("int_sink", sink_on_next, None, None)

        # Connect the source to the sink. You can also connect nodes by name
        seg.make_edge(node, sink)

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

    logger.info("srf pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    logger.info("srf pipeline complete: total_sum should be 6; total_sum={}".format(total_sum))


if (__name__ == "__main__"):
    run_pipeline()
