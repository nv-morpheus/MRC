import gc
import threading
import time
import weakref

from mrc.tests.utils import ObjCallingGC
from mrc.tests.utils import ObjUsingGil


class ThreadTest(threading.Thread):

    def _create_obs(self):
        ocg = ObjCallingGC()
        oug = ObjUsingGil()

        weakref.finalize(ocg, ocg.finalize)
        tls = threading.local()
        tls.oug = oug
        tls.ogc = ocg

    def run(self):
        print("Running thread", flush=True)
        self._create_obs()

        print("Sleeping", flush=True)
        time.sleep(1)
        print("Done sleeping", flush=True)


def test_gil_tls():
    t = ThreadTest()
    t.start()
    t.join()


def main():
    gc.disable()
    gc.set_debug(gc.DEBUG_STATS)
    test_gil_tls()


if __name__ == "__main__":
    main()
