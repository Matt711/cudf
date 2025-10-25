import random
import pyarrow as pa
import pylibcudf as plc
import numpy as np

np.random.seed(42)

rng = random.Random(2)
nrows = 300_000
key = [rng.choices(range(100), k=1)[0] for _ in range(nrows)]
value = [rng.choices(range(-100, 100), k=1)[0] for _ in range(nrows)]
tbl = plc.Table.from_arrow(
    pa.table(
        {
            "key0": pa.array(key, type=pa.int64()),
            "value": pa.array(value, type=pa.int64())
        }
    )
)
grouper = (
    plc.groupby.GroupBy(
        plc.Table([tbl.columns()[0]]),
        null_handling=plc.types.NullPolicy.INCLUDE,
        keys_are_sorted=plc.types.Sorted.NO,
        column_order=[plc.types.Order.ASCENDING],
        null_precedence=[plc.types.NullOrder.AFTER],
    )
)
group_keys, raw_tables = grouper.aggregate(
    [
        plc.groupby.GroupByRequest(
            tbl.columns()[1],
            [plc.aggregation.sum()]
        )
    ]
)
print("Total:", sum(value))
print("Grouped Total (hash):", sum(raw_tables[0].columns()[0].to_arrow().to_numpy()))