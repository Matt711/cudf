# Copyright (c) 2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

np = pytest.importorskip("numpy")
cp = pytest.importorskip("cupy")

CUPY_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.bool_,
]

NUMPY_DTYPES = [
    *CUPY_DTYPES,
    np.dtype("datetime64[s]"),
    np.dtype("datetime64[ms]"),
    np.dtype("datetime64[us]"),
    np.dtype("datetime64[ns]"),
    np.dtype("timedelta64[s]"),
    np.dtype("timedelta64[ms]"),
    np.dtype("timedelta64[us]"),
    np.dtype("timedelta64[ns]"),
]


@pytest.fixture(params=[1, 2, 3, 4], ids=lambda x: f"ndim={x}")
def shape(request):
    ndim = request.param
    shapes = {
        1: (6,),
        2: (2, 3),
        3: (2, 2, 3),
        4: (2, 2, 2, 3),
    }
    return shapes[ndim]


@pytest.fixture(params=CUPY_DTYPES, ids=repr)
def cp_array_and_np(request, shape):
    dtype = request.param
    size = np.prod(shape)
    if dtype == np.bool_:
        arr_np = np.array(
            [True, False] * ((size + 1) // 2), dtype=dtype
        ).reshape(shape)
    else:
        arr_np = np.arange(size, dtype=dtype).reshape(shape)
    return cp.asarray(arr_np), arr_np


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_array(request, shape):
    dtype = request.param
    size = np.prod(shape)
    if dtype == np.bool_:
        arr = np.array([True, False] * ((size + 1) // 2), dtype=dtype).reshape(
            shape
        )
    elif np.issubdtype(dtype, np.datetime64):
        unit = dtype.name.split("[")[-1][:-1]
        start = np.datetime64("2000-01-01", unit)
        step = np.timedelta64(1, unit)
        arr = np.arange(start, start + size * step, step, dtype=dtype).reshape(
            shape
        )
    else:
        arr = np.arange(size, dtype=dtype).reshape(shape)
    return arr


def test_from_cupy_array(cp_array_and_np):
    arr_cp, arr_np = cp_array_and_np
    arrow_type = pa.from_numpy_dtype(arr_np.dtype)
    for _ in range(len(arr_np.shape) - 1):
        arrow_type = pa.list_(arrow_type)
    expected = pa.array(arr_np.tolist(), type=arrow_type)

    result = plc.Column.from_array(arr_cp)
    assert_column_eq(expected, result)


def test_from_numpy_array(np_array):
    arr = np_array
    arrow_type = pa.from_numpy_dtype(arr.dtype)
    for _ in range(len(arr.shape) - 1):
        arrow_type = pa.list_(arrow_type)
    expected = pa.array(arr.tolist(), type=arrow_type)

    result = plc.Column.from_array(arr)
    assert_column_eq(expected, result)
