from contextlib import contextmanager
import time

@contextmanager
def stopwatch():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
