"""Does an advised column stay in the cuCascade cache across datasources? Wire bytes (NIC rx delta) per read."""
import os, sys, time, pylibcudf as plc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
path = os.environ.get("CC_TEST_PATH", "s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.0.parquet")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ev = os.environ.get("CC_EVICTION", "lru") == "lru"
cc.init(cache_bytes=8 << 30, endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16, lru=ev)
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
cols = ["l_extendedprice", "l_shipdate"]
for rep in range(3):
    r0 = rx(); t0 = time.time()
    ds = cc.CucascadeDatasource(path); src = plc.io.SourceInfo([ds])
    fm = plc.io.parquet_metadata.read_parquet_footers(src)[0]
    o = plc.io.parquet.ParquetReaderOptions.builder(src).build(); o.set_column_names(cols)
    rd = plc.io.experimental.HybridScanReader.from_metadata(plc.io.experimental.HybridScanMetadata.from_parquet_metadata(fm, o))
    rr = rd.all_column_chunks_byte_ranges(rd.all_row_groups(o), o)
    if os.environ.get("CC_ADVISE", "1") == "1": rc = ds.fadvise([r.offset for r in rr], [r.size for r in rr]); print("fadvise rc", rc)
    plc.io.parquet.read_parquet(o)
    dt = time.time() - t0; r1 = rx(); del ds, src
    print(f"rep{rep}: ranges={len(rr)} advised={sum(r.size for r in rr)/1e6:.0f} MB wire={(r1-r0)/1e6:.0f} MB {dt:.2f}s | {cc.cache_summary()}")
cc.shutdown()
