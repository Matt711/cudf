"""Which read shapes does the cuCascade cache keep? Each shape is read twice through fresh datasources (advised path, 1 MiB chunks)."""
import os, sys, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=4 << 30, endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16, column_policy=False,
        advise_min_bytes=1, initial_pools=32, block_bytes=1 << 20)
path = "s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.0.parquet"
M = 1 << 20
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def st():
    m = re.search(r"reads=(\d+) hits=(\d+) h2d=(\d+) miss=(\d+)", cc.cache_summary()); return tuple(map(int, m.groups()))
buf = np.empty(8 * M, dtype=np.uint8)
shapes = {"whole chunk [64M,65M)": (64 * M, M), "head half [66M,66.5M)": (66 * M, M // 2), "tail half [67.5M,68M)": (67 * M + M // 2, M // 2),
          "interior [69.25M,69.75M)": (69 * M + M // 4, M // 2), "straddle [71.75M,72.25M)": (71 * M + 3 * M // 4, M // 2),
          "3 MiB aligned [74M,77M)": (74 * M, 3 * M), "3 MiB unaligned [78.3M,81.3M)": (78 * M + 3 * M // 10, 3 * M)}
for name, (off, n) in shapes.items():
    out = []
    for rep in range(3):
        r0 = rx(); s0 = st(); ds = cc.CucascadeDatasource(path); ds.read_into(off, n, buf.ctypes.data); s1 = st(); del ds
        out.append(f"wire {(rx()-r0)/1e6:5.2f}MB hits+{s1[1]-s0[1]} miss+{s1[3]-s0[3]}")
    print(f"{name:32s}", " | ".join(out))
cc.shutdown()
