"""Pure I/O through the cuCascade datasource wrapper: N threads read column-chunk-sized ranges of lineitem objects (no decode).
usage: io_bench.py <threads> <range_MiB> [advise_min=1] [files=6]  env: BLK reactors CC_REACTORS CC_CONN"""
import os, sys, time, ctypes, numpy as np, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
NT = int(sys.argv[1]); RNG = int(float(sys.argv[2]) * (1 << 20)); AMB = int(sys.argv[3]) if len(sys.argv) > 3 else 1; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 6
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=int(float(os.environ.get("CC_CACHE_GB", "20")) * 2**30), endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=int(os.environ.get("CC_REACTORS", "16")),
        max_connections=int(os.environ.get("CC_CONN", "64")), advise_min_bytes=AMB, column_policy=False, block_bytes=int(os.environ.get("BLK", "1048576")), initial_pools=int(os.environ.get("POOLS", "4")))
paths = [f"s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.{i}.parquet" for i in range(NF)]
dss = [cc.CucascadeDatasource(p) for p in paths]
sizes = [3_100_000_000] * NF
jobs = [(i, o) for i in range(NF) for o in range(0, 2_000_000_000, RNG)]
bufs = [np.empty(RNG, dtype=np.uint8) for _ in range(NT)]
import threading; tl = threading.local(); idx = iter(range(NT)); lk = threading.Lock()
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def run(job):
    if not hasattr(tl, "b"):
        with lk: tl.b = bufs[next(idx)]
    i, o = job
    return dss[i].read_into(o, RNG, tl.b.ctypes.data)
r0 = rx(); t0 = time.time()
with cf.ThreadPoolExecutor(NT) as ex: tot = sum(ex.map(run, jobs))
dt = time.time() - t0
print(f"threads={NT} range={RNG>>20}MiB amb={AMB} blk={os.environ.get('BLK','1048576')}: {tot/1e9:.1f} GB in {dt:.1f}s = {tot/1e9/dt:.2f} GB/s  wire {(rx()-r0)/1e9:.1f} GB")
print(cc.cache_summary()[-120:])
del dss
cc.shutdown()
