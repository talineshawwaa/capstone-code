import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import Preprocessor


def make_sample_df():
    dates = pd.date_range("2022-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "date":              dates,
        "store_id":          "S001",
        "product_id":        "P0001",
        "category":          "Groceries",
        "region":            "North",
        "inventory_level":   np.random.uniform(50, 200, 50).astype(np.float32),
        "units_sold":        np.random.uniform(10, 100, 50).astype(np.float32),
        "units_ordered":     np.random.uniform(5, 80, 50).astype(np.float32),
        "demand_forecast":   np.random.uniform(10, 100, 50).astype(np.float32),
        "price":             np.random.uniform(10, 50, 50).astype(np.float32),
        "discount":          np.random.choice([0, 10, 20], 50).astype(np.float32),
        "weather_condition": "Sunny",
        "holiday_promotion": np.zeros(50, dtype=np.int8),
        "competitor_pricing": np.random.uniform(10, 50, 50).astype(np.float32),
        "seasonality":       "Summer",
    })
    return df


def test_fit_transform_returns_dataframe():
    df = make_sample_df()
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    result = pre.fit_transform(df)
    assert isinstance(result, pd.DataFrame)


def test_scaled_values_in_range():
    """Scaled values should be approximately in [0, 1] — allow float32 tolerance."""
    df = make_sample_df()
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    result = pre.fit_transform(df)
    assert result["units_sold"].min() >= -1e-5
    assert result["units_sold"].max() <= 1.0 + 1e-5


def test_no_negative_demand_after_clip():
    df = make_sample_df()
    df.loc[0, "units_sold"] = -5.0
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    result = pre.fit_transform(df)
    assert result["units_sold"].min() >= -1e-5


def test_inverse_transform_demand():
    df = make_sample_df()
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    pre.fit_transform(df)
    scaled = np.array([0.0, 0.5, 1.0])
    real   = pre.inverse_transform_demand(scaled, "S001", "P0001")
    assert real[0] <= real[1] <= real[2]


def test_scaler_saved_to_disk():
    df = make_sample_df()
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    pre.fit_transform(df)
    assert os.path.exists("/tmp/test_scalers/S001_P0001_scaler.pkl")


def test_transform_uses_fitted_scaler():
    df = make_sample_df()
    pre = Preprocessor(scaler_save_dir="/tmp/test_scalers/")
    pre.fit_transform(df)
    result = pre.transform(df.copy())
    assert result["units_sold"].min() >= -1e-5
    assert result["units_sold"].max() <= 1.0 + 1e-5
