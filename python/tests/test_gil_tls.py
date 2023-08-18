import threading
import weakref

import mrc
from mrc.tests.utils import ObjCallingGC

TLS = threading.local()


def test_gc_called_in_thread_finalizer():
    mrc.logging.log("Building pipeline")

    def source_gen():
        mrc.logging.log("source_gen")
        x = ObjCallingGC()
        weakref.finalize(x, x.finalize)
        TLS.x = x
        yield x

    def init_seg(builder: mrc.Builder):
        builder.make_source("souce_gen", source_gen)

    pipe = mrc.Pipeline()
    pipe.make_segment("seg1", init_seg)

    options = mrc.Options()
    executor = mrc.Executor(options)
    executor.register_pipeline(pipe)
    executor.start()
    executor.join()
