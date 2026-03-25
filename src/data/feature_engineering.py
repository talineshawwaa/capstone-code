import logging
import numpy as np
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, demand_history_length: int = 30):
        self.demand_history_length = demand_history_length

        # Will be populated by _encode_categoricals() — needed by other modules
        # to know which columns were one-hot encoded and their names.
        self.categorical_feature_columns: List[str] = []
        self.numeric_feature_columns: List[str] = []

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("FeatureEngineer: building features...")
        df = df.copy()

        df = self._add_lag_features(df)
        df = self._add_rolling_statistics(df)
        df = self._add_time_features(df)
        df = self._encode_categoricals(df)
        df = self._drop_nan_rows(df)

        logger.info(
            f"FeatureEngineer: complete. "
            f"Shape: {df.shape} | "
            f"Numeric features: {len(self.numeric_feature_columns)} | "
            f"Categorical features: {len(self.categorical_feature_columns)}"
        )
        return df

    # Step 1: Creating the Lag Features
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        def compute_lags(group):
            group = group.copy()
            # shift(1) moves values down by 1 row within the group.
            # Row T gets the value that was at row T-1.
            group["lag_1_units_sold"] = group["units_sold"].shift(1)
            group["lag_7_units_sold"] = group["units_sold"].shift(7)
            return group

        group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
        if group_cols:
            id_data = df[group_cols].copy()
            df = df.groupby(group_cols, group_keys=False).apply(compute_lags)
            df = df.reset_index(drop=True)
            for col in group_cols:
                if col not in df.columns:
                    df[col] = id_data[col].values
        else:
            df = compute_lags(df)
        logger.info("  ✓ Lag features added (lag_1, lag_7)")
        return df

    # Step 2: Creating the Rolling Statistics Features
    def _add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        def compute_rolling(group):
            group = group.copy()
            # Shift first, then roll — enforces the no-leakage rule.
            shifted = group["units_sold"].shift(1)
            group["roll_mean_7_units_sold"] = shifted.rolling(window=7, min_periods=1).mean()
            group["roll_std_7_units_sold"]  = shifted.rolling(window=7, min_periods=1).std().fillna(0)
            return group

        # Use store_id/product_id groupby only if both columns exist
        group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
        if group_cols:
            id_data = df[group_cols].copy()
            df = df.groupby(group_cols, group_keys=False).apply(compute_rolling)
            df = df.reset_index(drop=True)
            for col in group_cols:
                if col not in df.columns:
                    df[col] = id_data[col].values
        else:
            df = compute_rolling(df)
        logger.info("  ✓ Rolling statistics added (mean_7, std_7)")
        return df

    # Step 3: Extracting Time-Based Features
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"]       = df["date"].dt.month

        # Cyclical encoding for day of week (cycle length = 7)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Cyclical encoding for month (cycle length = 12)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        logger.info("  ✓ Time features added (day_of_week sin/cos, month sin/cos)")
        return df

    # Step 4: Encoding Categorical Variables
    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        id_cols = ["store_id", "product_id", "date"]
        saved = {col: df[col].values for col in id_cols if col in df.columns}

        categorical_cols = ["weather_condition", "seasonality", "category", "region"]
        df = pd.get_dummies(
            df,
            columns=categorical_cols,
            drop_first=False,
            dtype=float,        # use float so these columns match numeric types
        )

        for col, values in saved.items():
            if col not in df.columns:
                df[col] = values 
        # Record which columns were created — used by state_space.py to know
        # which columns to pull into the F_t^cat portion of the state vector.
        self.categorical_feature_columns = [
            c for c in df.columns
            if any(c.startswith(cat + "_") for cat in categorical_cols)
        ]

        # Record numeric feature columns — used by sequence_builder and LSTM.
        self.numeric_feature_columns = [
            "lag_1_units_sold",
            "lag_7_units_sold",
            "roll_mean_7_units_sold",
            "roll_std_7_units_sold",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "price",
            "discount",
            "competitor_pricing",
            "holiday_promotion",
        ]

        logger.info(
            f"  ✓ Categorical encoding: {len(self.categorical_feature_columns)} columns created"
        )
        return df

    # Dropping Missing Values After Feature Engineering

    def _drop_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        feature_cols = (
            ["lag_1_units_sold", "lag_7_units_sold",
             "roll_mean_7_units_sold", "roll_std_7_units_sold"]
        )
        df = df.dropna(subset=feature_cols).reset_index(drop=True)
        dropped = before - len(df)
        logger.info(f"  ✓ Dropped {dropped} NaN rows (from lag/rolling at series start)")
        return df