import os
import subprocess
import time 
import psutil
import functools
from contextlib import contextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from memory_profiler import profile as mem_profile  
from line_profiler import LineProfiler 

def profile_usage(func):
    @functools.wraps(func)
    def wrapper_profile_usage(*args, **kwargs):
        mem_before = psutil.virtual_memory().used
        result = func(*args, **kwargs)
        mem_after = psutil.virtual_memory().used
        mem_usage = mem_after - mem_before
        print(f"Memory usage: {mem_usage} bytes")
        return result, mem_usage

    return wrapper_profile_usage


def profile_latency(func):
    @functools.wraps(func)
    def wrapper_profile_latency(*args, **kwargs):
        profiler = LineProfiler()
        profiler.add_function(func)
        profiler.enable_by_count()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        profiler.disable_by_count()
        profiler.print_stats()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")
        return result, latency

    return wrapper_profile_latency