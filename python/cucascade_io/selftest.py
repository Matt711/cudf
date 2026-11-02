"""Integrity check: reads through the cuCascade datasource must equal standard libcudf reads. Credentials come from AWS_* env vars (never printed)."""
import os, sys, time, pyarrow as pa, pylibcudf as plc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
path = os.environ.get("CC_TEST_PATH", "s3://rapids-tpch/tpch-rs/scale-1/lineitem/part.0.parquet")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=int(float(os.environ.get("CC_CACHE_GB", "8")) * 2**30), endpoint=f"https://s3.{region}.amazonaws.com", region=region,
        access_key=os.environ["AWS_ACCESS_KEY_ID"], secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""))
def read(src, cols):
    o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([src])).build()
    if cols is not None: o.set_column_names(cols)
    r = plc.io.parquet.read_parquet(o)
    return pa.Table.from_arrays([c.to_arrow() for c in r.tbl.columns()], names=r.column_names(include_children=False))
colsets = [["l_extendedprice", "l_discount"], ["l_shipdate", "l_extendedprice"], ["l_orderkey", "l_partkey", "l_suppkey"], ["l_comment"], None]
ok = True
t0 = time.time(); ref = [read(path, c) for c in colsets]; print("reference reads %.1fs" % (time.time() - t0))
for rep in range(2):
    for c, r in zip(colsets, ref):
        t0 = time.time(); ds = cc.CucascadeDatasource(path); good = read(ds, c).equals(r); ok &= good
        print(f"rep{rep} {c}: equal={good} {time.time()-t0:.2f}s"); del ds
cc.shutdown()
print("ALL OK" if ok else "MISMATCH")
