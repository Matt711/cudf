# Integrity test for prefetch racing demand reads (late-claim path). Usage: CUDF_POLARS_BYTECACHE_GB=<cache MiB/1024> CUDF_POLARS_BYTECACHE_PATH=. python selftest_prefetch.py <cache MiB> <rounds> <threads>
"""Prefetch jobs and demand reads of the same columns run concurrently; every demand read must equal a standard read."""
import sys, threading, time, random, pyarrow as pa, pylibcudf as plc, _bytecache as bc, kvikio
from cudf_polars.utils import bytecache
path = "s3://rapids-tpch/tpch-rs/scale-1/lineitem/part.0.parquet"
n = kvikio.RemoteFile.open(path).nbytes()
def read(src, cols):
    o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([src])).build()
    if cols is not None:
        o.set_column_names(cols)
    tbl = plc.io.parquet.read_parquet(o)
    t = tbl.tbl
    return pa.Table.from_arrays([c.to_arrow() for c in t.columns()], names=tbl.column_names(include_children=False))
colsets = [["l_extendedprice", "l_discount"], ["l_shipdate", "l_extendedprice"], ["l_orderkey", "l_partkey", "l_suppkey"], ["l_comment"], ["l_quantity", "l_tax", "l_returnflag"]]
ref = [read(path, c) for c in colsets]
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 6
assert bytecache.enabled(), 'set CUDF_POLARS_BYTECACHE_GB (= cache MiB / 1024) and CUDF_POLARS_BYTECACHE_PATH'
ds0 = bc.CachedDatasource(path, n)
bytecache.register_layouts_from_sources([path], [ds0])
prefix = path.rsplit("/", 1)[0]
bad = []; slow = []
def demand(k):
    rnd = random.Random(k)
    for _ in range(3):
        i = rnd.randrange(len(colsets))
        t0 = time.time()
        got = read(bc.CachedDatasource(path, n), colsets[i])
        dt = time.time() - t0
        if not got.equals(ref[i]): bad.append((k, colsets[i]))
        slow.append(dt)
t_all = time.time()
for rnd_i in range(ROUNDS):
    bc.clear()
    for cols in colsets: bc.prefetch_columns(path, prefix, cols)     # asynchronous: returns while jobs run
    ths = [threading.Thread(target=demand, args=(rnd_i * 100 + k,)) for k in range(NT)]
    [t.start() for t in ths]; [t.join() for t in ths]
st = bc.get_stats()
print(f"cap {CAP} MiB, {ROUNDS} rounds x {NT} threads: mismatches={len(bad)} max demand read {max(slow):.2f}s mean {sum(slow)/len(slow):.2f}s total {time.time()-t_all:.1f}s")
print({k: st[k] for k in ('bytes_fetched', 'bytes_hit', 'bytes_prefetched', 'prefetch_jobs', 'blocks_evicted', 'waits_on_inflight')})
sys.exit(1 if bad else 0)
