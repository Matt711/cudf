"""Latency of opening a datasource (with and without a known size) and of a first small read, over several files."""
import os, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
import pylibcudf as plc
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=8 << 30, endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16, column_policy=False)
paths = [f"s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.{i}.parquet" for i in range(6)]
def timeit(f, n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter(); f(i); ts.append((time.perf_counter() - t0) * 1e3)
    return f"p50 {st.median(ts):6.1f} ms  max {max(ts):6.1f} ms"
print("open, no size, 1st pass :", timeit(lambda i: cc.CucascadeDatasource(paths[i % 6]), 6))
print("open, no size, 2nd pass :", timeit(lambda i: cc.CucascadeDatasource(paths[i % 6]), 6))
sizes = [cc.CucascadeDatasource(p).nbytes() for p in paths]
print("open, known size        :", timeit(lambda i: cc.CucascadeDatasource(paths[i % 6], sizes[i % 6]), 12))
def footer(i):
    ds = cc.CucascadeDatasource(paths[i % 6], sizes[i % 6]); plc.io.parquet_metadata.read_parquet_footers(plc.io.SourceInfo([ds]))
print("open(known size)+footer parse:", timeit(footer, 12))
cc.shutdown()
