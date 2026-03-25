import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_loader import ConfigLoader
from src.evaluation.comparison_report import ComparisonReport


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
    logger.info("GENERATING COMPARISON REPORT")
    logger.info("=" * 60)

    report = ComparisonReport(results_dir=cfg.base.paths.results)

    # Build master comparison table
    df = report.build()

    if df.empty:
        logger.error("No results found. Run scripts/run_scenarios.py first.")
        return

    # Print summary to console
    report.print_summary(df)

    # Save to CSV
    output_path = os.path.join(cfg.base.paths.results, "comparison_report.csv")
    report.save(df, output_path)

    logger.info(f"\nReport saved to: {output_path}")
    logger.info("Open this CSV for your thesis results chapter.")


if __name__ == "__main__":
    main()
