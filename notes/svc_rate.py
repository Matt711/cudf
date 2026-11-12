"""Standalone scan-service fetch rate: first GB of each lineitem file, engine settings. usage: svc_rate.py <label>"""
import os, sys, time
from cudf_polars_cucascade.scan_io import ScanIO
region = "us-east-2"
io = ScanIO(f"https://s3.{region}.amazonaws.com", region, os.environ.get("AWS_ACCESS_KEY_ID", ""),
            os.environ.get("AWS_SECRET_ACCESS_KEY", ""), os.environ.get("AWS_SESSION_TOKEN", ""),
            arena_bytes=32 << 30, reactors=int(os.environ.get("REACTORS", "32")), connections=int(os.environ.get("CONNS", "32")), footer_threads=32, inflight_bytes=4 << 30)
base = "s3://rapids-tpch/tpch-rs/scale-1000/lineitem/part.%d.parquet"
paths = [base % i for i in range(60)]
io.warmup(paths[0])
io.request_footers(paths)
sizes = [io.footer(p)[0] for p in paths]
chunk, per_task = 16 << 20, 1 << 30
for rep in range(int(os.environ.get("REPS", "2"))):
    s = io.open_stream()
    t0 = time.perf_counter(); f0 = io.fetched_bytes; ids = []
    for i, p in enumerate(paths):
        n = min(per_task, sizes[i] - (8 << 20))
        offs = list(range(0, n - chunk + 1, chunk))
        ids.append(io.submit(s, [p], [0] * len(offs), offs, [chunk] * len(offs)))
    for t in ids:
        io.wait(t); io.release(t)
    dt = time.perf_counter() - t0; gb = (io.fetched_bytes - f0) / 1e9
    io.close_stream(s)
    print(f"{sys.argv[1]} rep{rep}: {gb:.1f} GB in {dt:.2f} s = {gb/dt:.2f} GB/s", flush=True)
