# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest
from utils import assert_column_eq


@pytest.fixture(scope="module")
def input_col():
    arr = ["ab", "cde", "fgh"]
    return pa.array(arr)


@pytest.mark.parametrize("ngram", [2, 3])
@pytest.mark.parametrize("sep", ["_", "**", ","])
def test_generate_ngrams(input_col, ngram, sep):
    result = plc.nvtext.generate_ngrams.generate_ngrams(
        plc.interop.from_arrow(input_col),
        ngram,
        plc.interop.from_arrow(pa.scalar(sep)),
    )
    expected = pa.array([f"ab{sep}cde", f"cde{sep}fgh"])
    if ngram == 3:
        expected = pa.array([f"ab{sep}cde{sep}fgh"])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("ngram", [2, 3])
def test_generate_character_ngrams(input_col, ngram):
    result = plc.nvtext.generate_ngrams.generate_character_ngrams(
        plc.interop.from_arrow(input_col),
        ngram,
    )
    expected = pa.array([["ab"], ["cd", "de"], ["fg", "gh"]])
    if ngram == 3:
        expected = pa.array([[], ["cde"], ["fgh"]])
    assert_column_eq(result, expected)


@pytest.mark.parametrize("ngram", [2, 3])
def test_hash_character_ngrams(input_col, ngram):
    result = plc.nvtext.generate_ngrams.hash_character_ngrams(
        plc.interop.from_arrow(input_col),
        ngram,
    )
    pa_result = plc.interop.to_arrow(result)
    if ngram == 2:
        assert len(pa_result[0]) == 1
        assert len(pa_result[1]) == 2
        assert len(pa_result[2]) == 2
    else:
        assert len(pa_result[0]) == 0
        assert len(pa_result[1]) == 1
        assert len(pa_result[2]) == 1
