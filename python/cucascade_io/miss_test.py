"""One column from one file, repeated through the advising wrapper: wire bytes and cache miss/hit deltas per rep."""
import os, sys, re, time, pylibcudf as plc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _cucascade_io as cc
path = "s3://rapids-tpch/tpch-rs/scale-100/lineitem/part.0.parquet"
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
cc.init(cache_bytes=8 << 30, endpoint=f"https://s3.{region}.amazonaws.com", region=region, access_key=os.environ["AWS_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"], session_token=os.environ.get("AWS_SESSION_TOKEN", ""), rest_reactors=16, column_policy=False,
        advise_min_bytes=int(os.environ.get("AMB", "1")), block_bytes=int(os.environ.get("BLK", "1048576")))
def rx():
    for l in open("/proc/net/dev"):
        if l.split(":")[0].strip() == "enp39s0": return int(l.split(":")[1].split()[0])
def stats():
    m = re.search(r"global\[reads=(\d+) hits=(\d+) h2d=(\d+) miss=(\d+)", cc.cache_summary()); return tuple(map(int, m.groups()))
for cols in (["l_extendedprice"], ["l_discount"], ["l_extendedprice", "l_discount"]):
    for rep in range(3):
        r0 = rx(); s0 = stats()
        ds = cc.CucascadeDatasource(path)
        o = plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([ds])).build(); o.set_column_names(cols)
        plc.io.parquet.read_parquet(o); s1 = stats(); del ds
        print(f"{cols} rep{rep}: wire {(rx()-r0)/1e6:7.1f} MB  reads+{s1[0]-s0[0]} hits+{s1[1]-s0[1]} h2d+{s1[2]-s0[2]} miss+{s1[3]-s0[3]}")
cc.shutdown()
