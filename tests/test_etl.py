# ============================================================================
# Tests: ETLEngine
# ============================================================================

import pandas as pd
import pytest

from agentx.etl import DataCleaner, OcrEngine


def test_dedup_and_fill():
    df = pd.DataFrame({"name": ["A", "A", "b", None], "age": [30, 30, None, 45]})
    cleaner = DataCleaner(df).clean()
    out = cleaner.to_dataframe()
    assert len(out) == 3  # one duplicate removed
    assert out["age"].notna().all()  # missing age filled


def test_duplicate_count_in_report():
    df = pd.DataFrame({"x": [1, 1, 2]})
    cleaner = DataCleaner(df).clean()
    assert cleaner.report().duplicates_removed == 1
    assert cleaner.report().rows_before == 3
    assert cleaner.report().rows_after == 2


def test_type_inference_bool():
    df = pd.DataFrame({"flag": ["True", "False", "True"]})
    cleaner = DataCleaner(df).infer_dtypes()
    assert cleaner.to_dataframe()["flag"].dtype == bool


def test_outlier_capping():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 1000]})
    detected = DataCleaner(df).detect_outliers()
    assert 5 in detected["v"]  # index of 1000
    capped = DataCleaner(df).cap_outliers()
    assert capped.to_dataframe()["v"].max() < 1000


def test_missing_drop_strategy():
    df = pd.DataFrame({"a": [1, None, 3]})
    cleaner = DataCleaner(df).fill_missing(strategy="drop")
    assert len(cleaner.to_dataframe()) == 2


def test_trim_strings():
    df = pd.DataFrame({"name": ["  Alice  ", "Bob  "]})
    cleaner = DataCleaner(df).trim_strings()
    assert cleaner.to_dataframe()["name"].tolist() == ["Alice", "Bob"]


def test_cleaner_rejects_non_dataframe():
    with pytest.raises(TypeError):
        DataCleaner([1, 2, 3])


def test_ocr_engine_graceful():
    engine = OcrEngine(lang="eng")
    # Must never crash on import-time probe.
    assert isinstance(engine.is_available(), bool)
