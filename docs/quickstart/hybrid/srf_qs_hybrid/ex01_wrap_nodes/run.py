from srf_qs_hybrid.common import setup_logger
from srf_qs_hybrid.ex01_wrap_nodes import DataObjectNode
from srf_qs_hybrid.ex01_wrap_nodes import DataObjectSink
from srf_qs_hybrid.ex01_wrap_nodes import DataObjectSource

import srf

# Setup logging
logger = setup_logger(__file__)


def run_pipeline():

    def segment_init(seg: srf.Builder):

        # Create the source object
        src = DataObjectSource(seg, "src", count=3)

        node = DataObjectNode(seg, "node")

        seg.make_edge(src, node)

        sink = DataObjectSink(seg, "sink")

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

    print("srf pipeline starting...")

    # This will start the pipeline and return immediately
    executor.start()

    # Wait for the pipeline to exit on its own
    executor.join()

    print("srf pipeline complete")


if (__name__ == "__main__"):
    run_pipeline()
