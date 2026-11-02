"""How many bytes does a column read put on the wire through cuCascade, vs the column's compressed size? (NIC rx counter delta.)"""
import os, sys, time, pyarrow.parquet as pq, pylibcudf as plc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
path = os.environ.get("CC_TEST_PATH", "s3://rapids-tpch/tpch-rs/scale-10/lineitem/part.0.parquet")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=8 << 30, endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16)
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def read(src, cols):
    o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([src])).build(); o.set_column_names(cols)
    return plc.io.parquet.read_parquet(o)
for cols in (["l_extendedprice"], ["l_shipdate"], ["l_comment"], ["l_extendedprice", "l_discount", "l_quantity", "l_shipdate"]):
    for rep in range(2):
        r0 = rx(); t0 = time.time(); ds = cc.CucascadeDatasource(path); read(ds, cols); dt = time.time() - t0; r1 = rx(); del ds
        print(f"{cols} rep{rep}: wire {(r1-r0)/1e6:8.1f} MB  {dt:.2f}s")
