import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ScenarioBuilder:
    # Identifies store-product pairs for each evaluation scenario and returns their demand sequences

    def __init__(self, df_featured: pd.DataFrame, preprocessor, cfg):
        self.df       = df_featured
        self.pre      = preprocessor
        self.cfg      = cfg

    def build_scenario_a(self, n_pairs: int = 3) -> List[Dict]:
        # Scenario A - High volatility products
        logger.info("Building Scenario A: High-volatility products")
        return self._select_pairs_by_volatility(
            n_pairs=n_pairs,
            highest=True,
            scenario_name="A"
        )

    def build_scenario_b(self, n_pairs: int = 3) -> List[Dict]:
        # Scenario B — Stable, low-demand products.
        
        logger.info("Building Scenario B: Stable, low-demand products")
        return self._select_pairs_by_volatility(
            n_pairs=n_pairs,
            highest=False,
            scenario_name="B"
        )

    def build_scenario_c(
        self,
        train_store:   str,
        train_product: str,
        test_pairs:    Optional[List[Tuple[str, str]]] = None,
        n_test_pairs:  int = 20,
    ) -> List[Dict]:
        # Scenario C - Cross - Store generalization
        logger.info(
            f"Building Scenario C: Cross-store generalisation | "
            f"trained on {train_store}/{train_product}"
        )

        # Get all available pairs
        all_pairs = (
            self.df[["store_id", "product_id"]]
            .drop_duplicates()
            .apply(tuple, axis=1)
            .tolist()
        )

        # Exclude the training pair
        test_candidates = [
            (s, p) for s, p in all_pairs
            if not (s == train_store and p == train_product)
        ]

        if test_pairs is None:
            np.random.shuffle(test_candidates)
            test_pairs = test_candidates[:n_test_pairs]

        configs = []
        for store_id, product_id in test_pairs:
            config = self._build_pair_config(
                store_id=store_id,
                product_id=product_id,
                scenario_name="C",
                metadata={
                    "train_store":   train_store,
                    "train_product": train_product,
                    "note": "Cross-store generalisation test"
                }
            )
            if config:
                configs.append(config)

        return configs
    
    # PRIVATE HELPERS
    def _select_pairs_by_volatility(
        self,
        n_pairs:       int,
        highest:       bool,
        scenario_name: str,
    ) -> List[Dict]:
        # Selects pairs ranked by coefficient of variation
        
        # Compute CV per pair
        cv_records = []
        for (store, product), group in self.df.groupby(["store_id", "product_id"]):
            demand = group["units_sold"].values
            mean_d = np.mean(demand)
            std_d  = np.std(demand)
            cv     = std_d / mean_d if mean_d > 0 else 0
            cv_records.append({
                "store_id":   store,
                "product_id": product,
                "cv":         cv,
                "mean":       mean_d,
                "std":        std_d,
            })

        cv_df = pd.DataFrame(cv_records).sort_values(
            "cv", ascending=not highest
        ).reset_index(drop=True)

        selected = cv_df.head(n_pairs)
        logger.info(
            f"  Scenario {scenario_name}: selected {len(selected)} pairs | "
            f"CV range: [{selected['cv'].min():.3f}, {selected['cv'].max():.3f}]"
        )

        configs = []
        for _, row in selected.iterrows():
            config = self._build_pair_config(
                store_id=row["store_id"],
                product_id=row["product_id"],
                scenario_name=scenario_name,
                metadata={"cv": row["cv"], "mean_demand": row["mean"]}
            )
            if config:
                configs.append(config)

        return configs

    def _build_pair_config(
        self,
        store_id:      str,
        product_id:    str,
        scenario_name: str,
        metadata:      Dict = None,
    ) -> Optional[Dict]:
        # Builds a single scenario config dict for one store-product pair
        mask = (
            (self.df["store_id"] == store_id) &
            (self.df["product_id"] == product_id)
        )
        pair_df = self.df[mask].sort_values("date").reset_index(drop=True)

        if len(pair_df) < 50:
            logger.warning(
                f"  Skipping {store_id}/{product_id}: "
                f"only {len(pair_df)} rows (need ≥50)"
            )
            return None

        # Get demand in real units
        demand_scaled = pair_df["units_sold"].values.astype(np.float32)
        try:
            demand_real = self.pre.inverse_transform_demand(
                demand_scaled, store_id, product_id
            ).astype(np.float32)
        except Exception as e:
            logger.warning(f"  Could not inverse transform {store_id}/{product_id}: {e}")
            demand_real = demand_scaled * 500.0  # fallback approximation

        return {
            "store_id":    store_id,
            "product_id":  product_id,
            "scenario":    scenario_name,
            "demand_real": demand_real,
            "df_pair":     pair_df,
            "metadata":    metadata or {},
        }
