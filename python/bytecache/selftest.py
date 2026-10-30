# Integrity test for the bytecache extension (see DESIGN.md): cache reads must equal standard reads. Usage: python selftest.py <cache MiB> <threads>
"""Integrity test for the bytecache extension against an SF1 object on S3: cache reads must equal standard reads."""
import sys, threading, time, pyarrow as pa, pylibcudf as plc, _bytecache as bc, kvikio
path = "s3://rapids-tpch/tpch-rs/scale-1/lineitem/part.0.parquet"
n = kvikio.RemoteFile.open(path).nbytes()
def read(src, cols):
    o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([src])).build()
    o.set_column_names(cols)
    tbl = plc.io.parquet.read_parquet(o)
    t = tbl.tbl
    return pa.Table.from_arrays([c.to_arrow() for c in t.columns()], names=tbl.column_names(include_children=False))
colsets = [["l_extendedprice", "l_discount"], ["l_shipdate", "l_extendedprice"], ["l_orderkey", "l_partkey", "l_suppkey"], ["l_comment"]]
t0 = time.time(); ref = [read(path, c) for c in colsets]; print("reference reads %.1fs" % (time.time() - t0))
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 8   # MiB; 8 forces constant eviction on a 178 MB object, 0 disables caching
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 6
bc.configure(CAP << 20, 1 << 20, 4 << 20)
ds = bc.CachedDatasource(path, n)
ok = True
for rep in range(2):
    for c, r in zip(colsets, ref):
        t0 = time.time(); good = read(ds, c).equals(r); ok &= good
        print(f"rep{rep} {c}: equal={good} {time.time()-t0:.2f}s")
print("single-thread stats:", bc.get_stats())
res = []
def w(i):
    d = bc.CachedDatasource(path, n)
    for _ in range(3):
        for c, r in zip(colsets, ref): res.append(read(d, c).equals(r))
th = [threading.Thread(target=w, args=(i,)) for i in range(NT)]
[t.start() for t in th]; [t.join() for t in th]
print("concurrent: %d reads, all equal=%s" % (len(res), all(res)))
print("final stats:", bc.get_stats()); print("ALL OK" if ok and all(res) else "MISMATCH")
