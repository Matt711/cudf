"""Usable capacity of the cuCascade cache: read a fixed working set several times; per pass: wire bytes, hits, evictions.
usage: capacity_test.py <working_set_GB> [cache_GB=10] [evict_threshold=0.95]  (plain advised reads, 8 MiB ranges, 64 threads)"""
import os, sys, re, time, concurrent.futures as cf, threading, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
WS = float(sys.argv[1]); CAP = float(sys.argv[2]) if len(sys.argv) > 2 else 10; EV = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
RNG = 8 << 20; NT = 64
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=int(CAP * 2**30), endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16, column_policy=False,
        advise_min_bytes=1, evict_threshold=EV)
nf = 6; per = int(WS * 2**30 / nf)
dss = [cc.CucascadeDatasource(f"s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.{i}.parquet") for i in range(nf)]
jobs = [(i, o) for i in range(nf) for o in range(0, per - RNG + 1, RNG)]
bufs = [np.empty(RNG, dtype=np.uint8) for _ in range(NT)]; tl = threading.local(); lk = threading.Lock(); idx = iter(range(NT))
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def st():
    m = re.search(r"hits=(\d+) h2d=\d+ miss=(\d+) evictions=(\d+)", cc.cache_summary()); return tuple(map(int, m.groups()))
def run(j):
    if not hasattr(tl, "b"): tl.b = np.empty(RNG, dtype=np.uint8)
    return dss[j[0]].read_into(j[1], RNG, tl.b.ctypes.data)
tot = sum(1 for _ in jobs) * RNG
for p in range(3):
    r0 = rx(); s0 = st(); t0 = time.time()
    with cf.ThreadPoolExecutor(NT) as ex: sum(ex.map(run, jobs))
    s1 = st()
    print(f"ws={WS}GB cap={CAP}GB evict={EV} pass{p}: read {tot/1e9:.1f} GB wire {(rx()-r0)/1e9:5.1f} GB  hits+{s1[0]-s0[0]} miss+{s1[1]-s0[1]} evictions+{s1[2]-s0[2]}  {time.time()-t0:.1f}s")
del dss
cc.shutdown()
