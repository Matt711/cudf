"""Q7 vertical slice: lineitem probe chain run natively over a cold S3 fetch.

Untimed setup builds Q7's broadcast tables with the GPU engine. The timed part fetches lineitem's
footers and column chunks cold through ScanIO and runs both union branches' probe chains in the
native ProbePipeline. Reports when the last byte landed, when the pipeline finished, and validates
the aggregates against the expected answer.

usage: q7_native.py [threads]
"""

import os
import sys
import threading
import time
from datetime import date

import polars as pl
import pyarrow as pa

import pylibcudf as plc
import rmm.mr
from rmm.pylibrmm.stream import Stream

from cudf_polars.streaming import remote_io

THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROOT = "s3://rapids-tpch/tpch-rs/scale-1000"
NFILES = 60

mr = rmm.mr.CudaAsyncMemoryResource()
rmm.mr.set_current_device_resource(mr)


def to_plc(df: pl.DataFrame) -> plc.Table:
    return plc.Table.from_arrow(df.to_arrow(compat_level=pl.CompatLevel.oldest()))


# ------------------------------------------------------------------ untimed: build tables
t = time.perf_counter()
engine = pl.GPUEngine(executor="in-memory", memory_resource=mr, raise_on_fail=True)
scan = lambda name: pl.scan_parquet(f"{ROOT}/{name}/*.parquet")  # noqa: E731
nation = scan("nation").select("n_nationkey", "n_name").collect()
customer_orders = {}
for nat in ("FRANCE", "GERMANY"):
    customer_orders[nat] = (
        scan("customer")
        .select("c_custkey", "c_nationkey")
        .join(
            scan("nation").filter(pl.col("n_name") == nat).select("n_nationkey", "n_name"),
            left_on="c_nationkey",
            right_on="n_nationkey",
        )
        .join(scan("orders").select("o_orderkey", "o_custkey"), left_on="c_custkey", right_on="o_custkey")
        .select("o_orderkey", pl.col("n_name").alias("cust_nation"))
        .collect(engine=engine)
    )
supplier = scan("supplier").select("s_suppkey", "s_nationkey").collect(engine=engine)
builds = [
    to_plc(customer_orders["FRANCE"]),  # 0
    to_plc(customer_orders["GERMANY"]),  # 1
    to_plc(supplier),  # 2
    to_plc(nation.filter(pl.col("n_name") == "FRANCE")),  # 3
    to_plc(nation.filter(pl.col("n_name") == "GERMANY")),  # 4
]
build_keys = [[0], [0], [0], [0], [0]]
print(
    f"setup {time.perf_counter() - t:.1f}s build rows "
    f"{[b.num_rows() for b in builds]}",
    flush=True,
)
del customer_orders, supplier

# ------------------------------------------------------------------ the probe chains
# decoded chunk: [l_orderkey, l_suppkey, l_extendedprice, l_discount, l_shipdate]
SUB, MUL = int(plc.binaryop.BinaryOperator.SUB), int(plc.binaryop.BinaryOperator.MUL)
volume = [
    {"op": "column", "column": 0},
    {"op": "literal", "value": 1.0},
    {"op": "column", "column": 1},
    {"op": "binary", "binop": SUB, "lhs": 1, "rhs": 2},
    {"op": "binary", "binop": MUL, "lhs": 0, "rhs": 3},
]
year = [{"op": "column", "column": 2}, {"op": "year", "lhs": 0}]


def branch(cust_build: int, supp_nation_build: int) -> list[dict]:
    return [
        # -> [suppkey, price, discount, shipdate, cust_nation]
        {"op": "join", "build": cust_build, "probe_keys": [0], "keep_probe": [1, 2, 3, 4], "keep_build": [1]},
        # -> [price, discount, shipdate, cust_nation, s_nationkey]
        {"op": "join", "build": 2, "probe_keys": [0], "keep_probe": [1, 2, 3, 4], "keep_build": [1]},
        # -> [price, discount, shipdate, cust_nation, supp_nation]
        {"op": "join", "build": supp_nation_build, "probe_keys": [4], "keep_probe": [0, 1, 2, 3], "keep_build": [1]},
        # -> [..., volume, l_year]
        {"op": "compute", "exprs": [volume, year]},
        {"op": "groupby", "keys": [4, 3, 6], "sums": [5]},
    ]


branches = [branch(0, 4), branch(1, 3)]  # FRANCE customers / GERMANY suppliers, and the reverse

# ------------------------------------------------------------------ scan service and options
service = remote_io.configure_scan_io(
    arena_bytes=32768 << 20, reactors=32, connections=32, inflight_bytes=4096 << 20
)
service.warmup(f"{ROOT}/lineitem/part.0.parquet")
paths = [f"{ROOT}/lineitem/part.{i}.parquet" for i in range(NFILES)]
columns = ["l_orderkey", "l_suppkey", "l_extendedprice", "l_discount", "l_shipdate"]
stream = Stream()


def date_literal(d: date) -> tuple[plc.expressions.Literal, plc.Scalar]:
    s = plc.Scalar.from_arrow(pa.scalar(d, pa.date32()))
    return plc.expressions.Literal(s), s


lo, lo_s = date_literal(date(1995, 1, 1))
hi, hi_s = date_literal(date(1996, 12, 31))
ship = plc.expressions.ColumnNameReference("l_shipdate")
A = plc.expressions.ASTOperator
predicate = plc.expressions.Operation(
    A.LOGICAL_AND,
    plc.expressions.Operation(A.GREATER_EQUAL, ship, lo),
    plc.expressions.Operation(A.LESS_EQUAL, ship, hi),
)
options = (
    plc.io.parquet.ParquetReaderOptions.builder(plc.io.SourceInfo([paths[0]]))
    .decimal_width(plc.TypeId.DECIMAL128)
    .build()
)
options.set_column_names(columns)
options.set_filter(predicate)

from cudf_polars_cucascade.scan_io import ProbePipeline  # noqa: E402

pipeline = ProbePipeline(service, branches, builds, build_keys, THREADS)

# ------------------------------------------------------------------ timed: cold scan + probe
service.clear_footers()
fetched0 = service.fetched_bytes
t0 = time.perf_counter()
service.request_footers(paths)
scan_stream = service.open_stream()
done_planning = threading.Event()


def plan_all() -> None:
    for p in paths:
        task_id, empty, unsupported, too_big, staged, decoded = service.plan_task(
            scan_stream, [p], 0, 1, options, True, remote_io.MERGE_GAP_BYTES, stream
        )
        assert not (unsupported or too_big), (p, unsupported, too_big)
        if not empty:
            pipeline.submit(task_id)
    done_planning.set()


planner = threading.Thread(target=plan_all)
planner.start()
first_byte = last_byte = None
while True:
    waiting, inflight, queued, _free = service.snapshot()
    now = time.perf_counter() - t0
    if inflight and first_byte is None:
        first_byte = now
    if done_planning.is_set() and not (waiting or inflight or queued):
        last_byte = now
        break
    time.sleep(0.002)
planner.join()
results = pipeline.finish()
t_end = time.perf_counter() - t0
service.close_stream(scan_stream)
nbytes = service.fetched_bytes - fetched0
st = pipeline.stats()
print(
    f"threads {THREADS}: fetched {nbytes / 1e9:.1f} GB, first read {first_byte:.2f}s, "
    f"last byte {last_byte:.2f}s ({nbytes / 1e9 / last_byte:.1f} GB/s), "
    f"pipeline done {t_end:.2f}s, tail {t_end - last_byte:.2f}s",
    flush=True,
)
print({k: round(v, 3) for k, v in st.items()}, flush=True)

# ------------------------------------------------------------------ validate
names = ["supp_nation", "cust_nation", "l_year", "revenue"]
got = pl.concat(
    [
        pl.from_arrow(r.to_arrow(metadata=names))
        for r in results
    ]
).with_columns(pl.col("l_year").cast(pl.Int32)).sort("supp_nation", "cust_nation", "l_year")
expected = pl.read_parquet(f"{ROOT}/expected/q07.parquet").sort("supp_nation", "cust_nation", "l_year")
print(got)
ok = got.select(names[:3]).equals(expected.select(names[:3]).with_columns(pl.col("l_year").cast(pl.Int32))) and all(
    abs(a - b) <= 1e-6 * abs(b) for a, b in zip(got["revenue"], expected["revenue"], strict=True)
)
print("VALID" if ok else "MISMATCH", flush=True)
if not ok:
    print(expected)
os._exit(0)
