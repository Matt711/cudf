"""Time cuCascade cache creation (pins the whole cache) for a given size; usage: init_time.py <GiB> [pool_threads]"""
import os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
gib = float(sys.argv[1]); th = int(sys.argv[2]) if len(sys.argv) > 2 else 8
def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"): return int(l.split()[1]) / 1048576
a0 = avail(); t0 = time.time()
cc.init(cache_bytes=int(gib * 2**30), endpoint="https://s3.us-east-1.amazonaws.com", region="us-east-1", access_key="x", secret_key="x", pool_threads=th)
dt = time.time() - t0
print(f"cache {gib} GiB, pool_threads={th}: init took {dt:.1f} s; MemAvailable {a0:.0f} -> {avail():.0f} GB")
cc.shutdown()
