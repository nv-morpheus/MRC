import gc
import threading
import weakref

import mrc
from mrc.tests.utils import ObjCallingGC
from mrc.tests.utils import ObjUsingGil

TLS = threading.local()


class Holder:

    def __init__(self, obj):
        """Intentionally create a cycle to delay obj's destruction"""
        self.obj = obj
        self.cycle = self

    def __del__(self):
        mrc.logging.log("Holder.__del__")
        self.obj = None


class ThreadTest(threading.Thread):

    def _create_obs(self):
        TLS.h = Holder(ObjUsingGil())
        TLS.ocg = ObjCallingGC()
        # TLS.ocg = self.ocg
        weakref.finalize(TLS.ocg, TLS.ocg.finalize)

    def run(self):
        mrc.logging.log("Running thread")
        self._create_obs()
        mrc.logging.log("Thread complete")


def test_gil_tls():
    t = ThreadTest()
    t.start()
    t.join()
    mrc.logging.log("Thread joined")


def main():
    mrc.logging.init_logging(__name__)
    gc.disable()
    gc.set_debug(gc.DEBUG_STATS)
    test_gil_tls()
    mrc.logging.log("Exiting main")


if __name__ == "__main__":
    main()
