import os
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Columns that will be scaled to [0, 1].
# We exclude categorical columns (they get encoded later in feature_engineer.py)
# and the date column (it's not a feature, it's an index).
NUMERIC_COLUMNS_TO_SCALE = [
    "units_sold",
    "inventory_level",
    "units_ordered",
    "demand_forecast",
    "price",
    "discount",
    "competitor_pricing",
]

# Columns that must be present for the preprocessor to work.
REQUIRED_COLUMNS = [
    "date", "store_id", "product_id", "units_sold",
    "inventory_level", "price", "discount",
    "weather_condition", "holiday_promotion",
    "competitor_pricing", "seasonality",
]


class Preprocessor:
    """
    Cleans and normalises the loaded DataFrame.

    Maintains one MinMaxScaler per (store_id, product_id) pair so that
    each time series is scaled independently. The scalers are saved to disk
    so that predictions can be inverse-transformed back to real units.
    """

    def __init__(self, scaler_save_dir: str = "models/lstm/scalers/"):
        self.scaler_save_dir = scaler_save_dir
        os.makedirs(scaler_save_dir, exist_ok=True)

        # Dict mapping (store_id, product_id) → fitted MinMaxScaler.
        self.scalers: Dict[Tuple[str, str], MinMaxScaler] = {}

    # PUBLIC METHODS

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cleans the DataFrame and fits+applies scalers. Call this on training data.
        logger.info("Preprocessor: fit_transform started")
        df = self._validate(df)
        df = self._clip_negative_demand(df)
        df = self._remove_duplicates(df)
        df = self._fill_date_gaps(df)
        df = self._fit_and_scale(df)
        self._save_scalers()
        logger.info("Preprocessor: fit_transform complete")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Cleans and applies already-fitted scalers. Call this on val/test data.
        
        if not self.scalers:
            raise RuntimeError(
                "No scalers loaded. Call load_scalers() before transform()."
            )
        logger.info("Preprocessor: transform started")
        df = self._validate(df)
        df = self._clip_negative_demand(df)
        df = self._remove_duplicates(df)
        df = self._fill_date_gaps(df)
        df = self._apply_scale(df)
        logger.info("Preprocessor: transform complete")
        return df

    def inverse_transform_demand(
        self, scaled_values: np.ndarray, store_id: str, product_id: str
    ) -> np.ndarray:
        # Converts scaled demand predictions back to real unit values.
        key = (store_id, product_id)
        if key not in self.scalers:
            raise KeyError(f"No scaler found for {key}. Was fit_transform called?")

        scaler = self.scalers[key]
        units_sold_idx = NUMERIC_COLUMNS_TO_SCALE.index("units_sold")

        # MinMaxScaler.inverse_transform expects shape (N, n_features).
        # We build a dummy array of zeros and put our values in the correct column,
        # then extract just that column after inverse transforming.
        original_shape = scaled_values.shape
        flat = scaled_values.reshape(-1, 1)
        dummy = np.zeros((len(flat), len(NUMERIC_COLUMNS_TO_SCALE)))
        dummy[:, units_sold_idx] = flat[:, 0]
        inverted = scaler.inverse_transform(dummy)
        return inverted[:, units_sold_idx].reshape(original_shape)

    def load_scalers(self) -> None:
        # Loads all saved scaler pickle files from scaler_save_dir into memory.
        loaded = 0
        for filename in os.listdir(self.scaler_save_dir):
            if filename.endswith(".pkl"):
                path = os.path.join(self.scaler_save_dir, filename)
                with open(path, "rb") as f:
                    key, scaler = pickle.load(f)
                self.scalers[key] = scaler
                loaded += 1
        logger.info(f"Loaded {loaded} scalers from {self.scaler_save_dir}")

    # PRIVATE HELPERS
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Checks all required columns are present
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        return df.copy()  # always work on a copy, never mutate the input

    def _clip_negative_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clips units_sold to [0, ∞). Demand is always non-negative.
        n_negative = (df["units_sold"] < 0).sum()
        if n_negative > 0:
            logger.warning(f"  Clipping {n_negative} negative demand values to 0")
        df["units_sold"] = df["units_sold"].clip(lower=0)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        # Removes rows where (store_id, product_id, date) is duplicated.
        before = len(df)
        df = df.drop_duplicates(subset=["store_id", "product_id", "date"], keep="first")
        removed = before - len(df)
        if removed > 0:
            logger.warning(f"  Removed {removed} duplicate rows")
        return df

    def _fill_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensures every (store, product) pair has a continuous daily time series with no missing dates.

        groups = []
        for (store, product), group in df.groupby(["store_id", "product_id"]):
            group = group.set_index("date")
            # reindex to a complete daily date range
            full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
            group = group.reindex(full_range)
            n_filled = group["units_sold"].isna().sum()
            if n_filled > 0:
                logger.warning(
                    f"  {store}/{product}: forward-filling {n_filled} missing dates"
                )
            group = group.ffill()
            # Restore store_id and product_id (they become NaN after reindex)
            group["store_id"] = store
            group["product_id"] = product
            group = group.reset_index().rename(columns={"index": "date"})
            groups.append(group)

        return pd.concat(groups, ignore_index=True).sort_values(
            ["store_id", "product_id", "date"]
        ).reset_index(drop=True)

    def _fit_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Fits one MinMaxScaler per (store_id, product_id) group on the numeric columns and applies it.
        scaled_groups = []
        for (store, product), group in df.groupby(["store_id", "product_id"]):
            group = group.copy()
            scaler = MinMaxScaler(feature_range=(0, 1))
            group[NUMERIC_COLUMNS_TO_SCALE] = scaler.fit_transform(
                group[NUMERIC_COLUMNS_TO_SCALE]
            )
            self.scalers[(store, product)] = scaler
            scaled_groups.append(group)

        logger.info(f"  Fitted {len(self.scalers)} scalers")
        return pd.concat(scaled_groups, ignore_index=True).sort_values(
            ["store_id", "product_id", "date"]
        ).reset_index(drop=True)

    def _apply_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        # Applies already-fitted scalers (from self.scalers) to the DataFrame.
        scaled_groups = []
        for (store, product), group in df.groupby(["store_id", "product_id"]):
            key = (store, product)
            if key not in self.scalers:
                raise KeyError(
                    f"No fitted scaler for ({store}, {product}). "
                    f"Was fit_transform called on training data first?"
                )
            group = group.copy()
            group[NUMERIC_COLUMNS_TO_SCALE] = self.scalers[key].transform(
                group[NUMERIC_COLUMNS_TO_SCALE]
            )
            scaled_groups.append(group)

        return pd.concat(scaled_groups, ignore_index=True).sort_values(
            ["store_id", "product_id", "date"]
        ).reset_index(drop=True)

    def _save_scalers(self) -> None:
        # Saves each fitted scaler as a separate pickle file.
        for (store, product), scaler in self.scalers.items():
            filename = f"{store}_{product}_scaler.pkl"
            path = os.path.join(self.scaler_save_dir, filename)
            with open(path, "wb") as f:
                pickle.dump(((store, product), scaler), f)
        logger.info(f"  Saved {len(self.scalers)} scalers to {self.scaler_save_dir}")