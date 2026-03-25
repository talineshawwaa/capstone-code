import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


class ComparisonReport:
    # Builds the master KPI comparison table across all agents and scenarios

    def __init__(self, results_dir: str = "outputs/results/"):
        self.results_dir = results_dir

    def build(self) -> pd.DataFrame:
        # Reads all result JSON files and builds a master comparison DataFrame
        rows = []

        for scenario_label in ["A", "B", "C"]:
            result_path = os.path.join(
                self.results_dir,
                f"scenario_{scenario_label.lower()}",
                "results.json"
            )

            if not os.path.exists(result_path):
                logger.warning(f"No results found for Scenario {scenario_label}: {result_path}")
                continue

            with open(result_path, "r") as f:
                scenario_results = json.load(f)

            for pair_key, agent_results in scenario_results.items():
                for agent_name, kpis in agent_results.items():
                    row = {
                        "Scenario":     f"Scenario {scenario_label}",
                        "Pair":         pair_key,
                        "Agent":        agent_name,
                        "Cumulative Reward":     kpis.get("total_cumulative_reward", 0),
                        "Holding Cost":          kpis.get("total_holding_cost", 0),
                        "Lost Sales (units)":    kpis.get("total_lost_sales", 0),
                        "Ordering Quantity":     kpis.get("total_ordering_quantity", 0),
                        "Service Level (%)":     kpis.get("service_level", 0) * 100,
                        "Avg Inventory":         kpis.get("avg_inventory", 0),
                        "Total Cost":            kpis.get("total_cost", 0),
                    }
                    rows.append(row)

        if not rows:
            logger.error("No results found. Run run_scenarios.py first.")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    def save(self, df: pd.DataFrame, output_path: str = "outputs/results/comparison_report.csv") -> None:
        # Saves the comparison DataFrame to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Comparison report saved to {output_path}")

    def print_summary(self, df: pd.DataFrame) -> None:
        # Prints a formatted summary table to the console
        if df.empty:
            print("No results to display.")
            return

        print("\n" + "="*80)
        print("MASTER COMPARISON REPORT")
        print("="*80)

        kpi_cols = [
            "Cumulative Reward", "Holding Cost", "Lost Sales (units)",
            "Ordering Quantity", "Service Level (%)", "Total Cost"
        ]

        for scenario in df["Scenario"].unique():
            print(f"\n{scenario}")
            print("-" * 60)

            scenario_df = df[df["Scenario"] == scenario]
            summary = scenario_df.groupby("Agent")[kpi_cols].mean().round(2)
            print(summary.to_string())

        print("\n" + "="*80)
        print("OVERALL AVERAGE ACROSS ALL SCENARIOS")
        print("="*80)
        overall = df.groupby("Agent")[kpi_cols].mean().round(2)
        print(overall.to_string())
