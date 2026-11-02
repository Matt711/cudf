"""Parallel column reads through cuCascade: advised (per-read fadvise+prefetch) vs plain. usage: par_test.py <advise_min_bytes> [files=8] [threads=4]"""
import os, sys, time, concurrent.futures as cf, pylibcudf as plc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
amb = int(sys.argv[1]); NF = int(sys.argv[2]) if len(sys.argv) > 2 else 8; NT = int(sys.argv[3]) if len(sys.argv) > 3 else 4
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=int(float(os.environ.get("CC_CACHE_GB", "20")) * 2**30), endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=int(os.environ.get("CC_REACTORS", "16")), advise_min_bytes=amb)
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def one(i):
    ds = cc.CucascadeDatasource(f"s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.{i}.parquet")
    o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([ds])).build(); o.set_column_names(["l_extendedprice", "l_shipdate", "l_discount", "l_quantity"])
    return plc.io.parquet.read_parquet(o).tbl.num_rows()
for rep in range(2):
    r0 = rx(); t0 = time.time()
    with cf.ThreadPoolExecutor(NT) as ex: n = sum(ex.map(one, range(NF)))
    dt = time.time() - t0
    print(f"advise_min={amb} rep{rep}: {dt:.2f}s wire={(rx()-r0)/1e9:.2f} GB rate={(rx()-r0)/1e9/dt:.2f} GB/s rows={n} | {cc.cache_summary()[:90]}")
