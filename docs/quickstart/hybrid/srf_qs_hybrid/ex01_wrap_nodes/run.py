from srf_qs_hybrid.common import setup_logger
from srf_qs_hybrid.ex01_wrap_nodes import MyDataObjectNode
from srf_qs_hybrid.ex01_wrap_nodes import MyDataObjectSink
from srf_qs_hybrid.ex01_wrap_nodes import MyDataObjectSource

import srf

# Setup logging
logger = setup_logger(__file__)


def run_pipeline():

    def segment_init(seg: srf.Builder):

        # Create the source object
        src = MyDataObjectSource(seg, "src", count=3)

        node = MyDataObjectNode(seg, "node")

        seg.make_edge(src, node)

        sink = MyDataObjectSink(seg, "sink")

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

    logger.info("srf pipeline complete")


if (__name__ == "__main__"):
    run_pipeline()
