import os
import pandas as pd
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# Column Remapping, so that they all have the same naming
column_remapping = {
    "Date":               "date",
    "Store ID":           "store_id",
    "Product ID":         "product_id",
    "Category":           "category",
    "Region":             "region",
    "Inventory Level":    "inventory_level",
    "Units Sold":         "units_sold",        
    "Units Ordered":      "units_ordered",
    "Demand Forecast":    "demand_forecast",   
    "Price":              "price",
    "Discount":           "discount",
    "Weather Condition":  "weather_condition",
    "Holiday/Promotion":  "holiday_promotion",
    "Competitor Pricing": "competitor_pricing",
    "Seasonality":        "seasonality",
}

# DTYPE Mapping for memory optimization
dtype_map = {
    "store_id":           "string",
    "product_id":         "string",
    "category":           "string",
    "region":             "string",
    "inventory_level":    "float32",
    "units_sold":         "float32",
    "units_ordered":      "float32",
    "demand_forecast":    "float32",
    "price":              "float32",
    "discount":           "float32",
    "weather_condition":  "string",
    "holiday_promotion":  "int8",    # binary 0/1 flag
    "competitor_pricing": "float32",
    "seasonality":        "string",
}

class DataLoader:
    def __init__(self, filepath:str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Raw data file not found: {filepath}\n"
                f"Place your CSV in the data/raw/ folder."
            )
        self.filepath = filepath
        logger.info(f"DataLoader initialised — file: {filepath}")

    def load(self, stores: Optional[List[str]] = None, products: Optional[List[str]] = None,) -> pd.DataFrame:
        logger.info("Loading raw CSV...")

        # Step 1: read CSV 
        df = pd.read_csv(self.filepath, low_memory=False)
        logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Step 2: rename columns to snake_case
        df = df.rename(columns=column_remapping)

        # Step 3: parse date column
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

        # Step 4: Enforce dtypes
        for col, dtype in dtype_map.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        
        # Step 5: filtering stores and products
        if stores is not None:
            before = len(df)
            df = df[df["store_id"].isin(stores)]
            logger.info(f"  Filtered to stores {stores}: {before:,} → {len(df):,} rows")
            
        if products is not None:
            before = len(df)
            df = df[df["product_id"].isin(products)]
            logger.info(f"  Filtered to products {products}: {before:,} → {len(df):,} rows")

        # Sorting the DataFrame by store_id, product_id, and date ensures that sequences 
        # are built in the correct order.
        df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

        logger.info(
            f"  Final shape: {df.shape} | "
            f"Stores: {df['store_id'].nunique()} | "
            f"Products: {df['product_id'].nunique()} | "
            f"Date range: {df['date'].min().date()} → {df['date'].max().date()}"
        )

        return df
    
    # This method is used by the SequenceBuilder to get the unique store-product pairs in the dataset.
    def get_store_product_pairs(self, df: pd.DataFrame) -> List:
        pairs = (
            df[["store_id", "product_id"]]
            .drop_duplicates()
            .sort_values(["store_id", "product_id"])
            .apply(tuple, axis=1)
            .tolist()
        )

        logger.info(f"Found {len(pairs):,} unique store-product pairs.")
        return pairs