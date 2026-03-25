import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class SequenceBuilder:
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 7, train_split: float = 0.70, val_split: float = 0.15):
        self.sequence_length  = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train_split      = train_split
        self.val_split        = val_split
        self.test_split       = 1.0 - train_split - val_split

        if self.test_split <= 0:
            raise ValueError(f"train_split ({train_split}) + val_split ({val_split}) must be < 1.0")
    
    def build(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if "units_sold" not in df.columns:
            raise ValueError("DataFrame must contain 'units_sold' column as forecast target.")

        if "units_sold" not in feature_cols:
            # Always include units_sold in features so the LSTM sees past demand
            feature_cols = ["units_sold"] + feature_cols

        all_X = {"train": [], "val": [], "test": []}
        all_y = {"train": [], "val": [], "test": []}

        group_cols = [c for c in ["store_id", "product_id"] if c in df.columns]
        if group_cols:
            pairs = df.groupby(group_cols)
        else:
            # Treat the whole DataFrame as one pair
            pairs = [("single", df)]
        logger.info(f"SequenceBuilder: processing {len(list(pairs))} store-product pairs...")
        if group_cols:
            pairs = df.groupby(group_cols)

        for key, group in pairs:
            store   = key[0] if isinstance(key, tuple) else "single"
            product = key[1] if isinstance(key, tuple) and len(key) > 1 else "single"
            group = group.sort_values("date").reset_index(drop=True)

            # Extract the 2D feature matrix for this group: shape (T, n_features)
            features = group[feature_cols].values.astype(np.float32)

            # Extract the 1D demand target for this group: shape (T,)
            # units_sold is always the column we forecast.
            demand = group["units_sold"].values.astype(np.float32)

            # Build sliding window sequences for this group
            X_seq, y_seq = self._build_sequences(features, demand)

            if len(X_seq) == 0:
                logger.warning(
                    f"  Skipping {store}/{product}: not enough data for even one window "
                    f"(need {self.sequence_length + self.forecast_horizon} days, "
                    f"got {len(group)})"
                )
                continue

            # Split sequences chronologically
            n = len(X_seq)
            train_end = int(n * self.train_split)
            val_end   = int(n * (self.train_split + self.val_split))

            all_X["train"].append(X_seq[:train_end])
            all_y["train"].append(y_seq[:train_end])

            all_X["val"].append(X_seq[train_end:val_end])
            all_y["val"].append(y_seq[train_end:val_end])

            all_X["test"].append(X_seq[val_end:])
            all_y["test"].append(y_seq[val_end:])

        # Concatenate all pairs into single arrays
        result = {}
        for split in ["train", "val", "test"]:
            if not all_X[split]:
                raise RuntimeError(f"No sequences were built for the '{split}' split.")
            X = np.concatenate(all_X[split], axis=0)
            y = np.concatenate(all_y[split], axis=0)
            result[split] = (X, y)
            logger.info(f"  {split:5s}: X={X.shape}, y={y.shape}")

        return result

    def build_single_pair(self, df: pd.DataFrame, store_id: str, product_id: str, feature_cols: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        mask = (df["store_id"] == store_id) & (df["product_id"] == product_id)
        pair_df = df[mask].copy()
        if len(pair_df) == 0:
            raise ValueError(f"No data found for ({store_id}, {product_id})")
        return self.build(pair_df, feature_cols)
    
    def _build_sequences(self, features: np.ndarray, demand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(features)
        min_required = self.sequence_length + self.forecast_horizon

        if T < min_required:
            return np.array([]), np.array([])

        X_list = []
        y_list = []

        # i is the index of the LAST timestep in the input window.
        # The input window covers [i - seq_len + 1 : i + 1].
        # The output covers [i + 1 : i + 1 + horizon].
        for i in range(self.sequence_length - 1, T - self.forecast_horizon):
            window_start = i - self.sequence_length + 1
            window_end   = i + 1                          # exclusive
            target_start = i + 1
            target_end   = i + 1 + self.forecast_horizon  # exclusive

            X_list.append(features[window_start:window_end])   # (seq_len, n_feat)
            y_list.append(demand[target_start:target_end])      # (horizon,)

        X = np.array(X_list, dtype=np.float32)  # (n_samples, seq_len, n_feat)
        y = np.array(y_list, dtype=np.float32)  # (n_samples, horizon)

        return X, y


