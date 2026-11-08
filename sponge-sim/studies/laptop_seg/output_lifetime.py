"""Explicit output lifetime for synchronous CPU function timing.

Release occurs AFTER wall/CPU clock endpoints, BEFORE the next invocation.
No operation of the function is moved outside its measured interval.
"""
import gc
import time


def measure(fn,capture_cpu=False,keep_output=True):
    enabled=gc.isenabled();gc.disable()
    try:
        c0=time.process_time_ns() if capture_cpu else None
        t0=time.perf_counter_ns();result=fn();ns=time.perf_counter_ns()-t0
        cpu_ns=None if c0 is None else time.process_time_ns()-c0
        if not keep_output:
            del result
            result=None
        return ns,t0,result,cpu_ns
    finally:
        if enabled:gc.enable()
