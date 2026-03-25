import os
import sys
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.evaluation.comparison_report import ComparisonReport
from src.visualizations.comparison_plots import (
    plot_kpi_comparison,
    plot_cost_breakdown,
    plot_service_level_comparison,
)


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    cfg = ConfigLoader()
    setup_logging(cfg.base.paths.logs)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("GENERATING THESIS FIGURES")
    logger.info("=" * 60)

    plots_dir = cfg.base.paths.plots

    # Load comparison results
    report = ComparisonReport(results_dir=cfg.base.paths.results)
    df     = report.build()

    if df.empty:
        logger.error("No results found. Run scripts/run_scenarios.py first.")
        return

    logger.info(f"Loaded results: {len(df)} rows across {df['Agent'].nunique()} agents")

    # 1. KPI comparison bar charts (one per KPI, faceted by scenario)
    logger.info("Generating KPI comparison charts...")
    plot_kpi_comparison(df, save_dir=plots_dir, show=False)

    # 2. Cost breakdown stacked bar (one per scenario)
    logger.info("Generating cost breakdown charts...")
    for scenario in df["Scenario"].unique():
        plot_cost_breakdown(df, scenario=scenario, save_dir=plots_dir, show=False)

    # 3. Service level across scenarios
    logger.info("Generating service level comparison...")
    plot_service_level_comparison(df, save_dir=plots_dir, show=False)

    # 4. Print final summary table
    report.print_summary(df)

    logger.info(f"\nAll figures saved to: {plots_dir}")

if __name__ == "__main__":
    main()
