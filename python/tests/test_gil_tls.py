import gc
import threading
import weakref

from mrc.tests.utils import ObjCallingGC
from mrc.tests.utils import ObjUsingGil


class Holder:

    def __init__(self, obj):
        """Intentionally create a cycle to delay obj's destruction"""
        self.obj = obj
        self.cycle = self

    def __del__(self):
        print("Holder.__del__", flush=True)
        self.obj = None


class ThreadTest(threading.Thread):

    def _create_obs(self):
        self.h = Holder(ObjUsingGil())
        self.ocg = ObjCallingGC()
        weakref.finalize(self.ocg, self.ocg.finalize)

    def run(self):
        print("Running thread", flush=True)
        self._create_obs()
        print("Thread complete", flush=True)


def test_gil_tls():
    t = ThreadTest()
    t.start()
    t.join()
    print("Thread joined, dereferencing thread", flush=True)
    t = None


def main():
    gc.disable()
    gc.set_debug(gc.DEBUG_STATS)
    test_gil_tls()
    print("Exiting main", flush=True)


if __name__ == "__main__":
    main()
