"""Fetch-rate microbenchmark (no GPU work): is the ~5 GB/s ceiling in the cache's read path or beyond it?
  A: plain kvikio, NT threads reading 4 MB ranges of the first GB of 60 lineitem objects (about 60 GB)
  B: the bytecache prefetch path over the q1 columns of the same 60 objects (about 57 GB)"""
import sys, time, threading, random, numpy as np, kvikio
mode, NT = sys.argv[1], int(sys.argv[2])
base = "s3://rapids-tpch/tpch-rs/scale-1000/lineitem/part.%d.parquet"
paths = [base % i for i in range(60)]
if mode == "A":
    files = [kvikio.RemoteFile.open(p) for p in paths]
    jobs = [(f, off) for f in files for off in range(0, 1 << 30, 4 << 20)]
    random.seed(1); random.shuffle(jobs)
    it = iter(jobs); lock = threading.Lock(); done = [0]
    def w():
        buf = np.empty(4 << 20, dtype=np.uint8)
        while True:
            with lock:
                j = next(it, None)
            if j is None: return
            j[0].read(buf, 4 << 20, j[1]); done[0] += 4 << 20
    t0 = time.time(); th = [threading.Thread(target=w) for _ in range(NT)]
    [t.start() for t in th]; [t.join() for t in th]; dt = time.time() - t0
    print(f"A plain kvikio, {NT} threads: {done[0]/1e9:.1f} GB in {dt:.1f} s = {done[0]/1e9/dt:.1f} GB/s")
else:
    from cudf_polars.utils import bytecache
    import _bytecache as bc
    bytecache.enabled()
    from concurrent.futures import ThreadPoolExecutor
    def reg(p):
        s = bc.CachedDatasource(p, 0); bytecache.register_layouts_from_sources([p], [s])
    t0 = time.time()
    with ThreadPoolExecutor(32) as ex: list(ex.map(reg, paths))
    print(f"registered {len(paths)} layouts in {time.time()-t0:.1f} s")
    cols = ["l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate"]
    t0 = time.time(); tot = sum(bc.prefetch_columns(p, p.rsplit("/", 1)[0], cols) for p in paths)
    last = -1; stable = 0
    while True:
        time.sleep(0.25); st = bc.get_stats(); res = st["blocks_resident"]
        if res * (1 << 18) >= 0.99 * tot or time.time() - t0 > 90: break
    dt = time.time() - t0
    print(f"B bytecache prefetch: {tot/1e9:.1f} GB in {dt:.1f} s = {tot/1e9/dt:.1f} GB/s (resident {res*(1<<18)/1e9:.1f} GB)")
