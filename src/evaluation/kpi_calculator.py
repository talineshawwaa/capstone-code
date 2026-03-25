import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class KPICalculator:
    # Computes KPI metrics from episode step data collected by the backtester

    def compute(self, step_records: List[Dict]) -> Dict[str, float]:
        # Computes all KPIs
        if not step_records:
            return self._empty_kpis()

        # Extract arrays from step records
        rewards         = np.array([s["reward"]         for s in step_records])
        holding_costs   = np.array([s["holding_cost"]   for s in step_records])
        lost_sales      = np.array([s["lost_sales"]      for s in step_records])
        order_qtys      = np.array([s["order_quantity"]  for s in step_records])
        demands         = np.array([s["demand"]          for s in step_records])
        inventories     = np.array([s["inventory"]       for s in step_records])

        # 1. Total Cumulative Reward 
        total_cumulative_reward = float(np.sum(rewards))

        # 2. Total Inventory Holding Cost 
        total_holding_cost = float(np.sum(holding_costs))

        # 3. Total Lost Sales 
        total_lost_sales = float(np.sum(lost_sales))

        # 4. Total Ordering Quantity 
        total_ordering_quantity = float(np.sum(order_qtys))

        # 5. Service Level (Fill Rate) 
        total_demand = float(np.sum(demands))
        if total_demand > 0:
            service_level = (total_demand - total_lost_sales) / total_demand
            service_level = float(np.clip(service_level, 0.0, 1.0))
        else:
            service_level = 1.0

        # 6. Average Daily Inventory
        avg_inventory = float(np.mean(inventories))

        #  7. Total Operational Cost
        total_cost = float(-np.sum(rewards))

        kpis = {
            "total_cumulative_reward":    total_cumulative_reward,
            "total_holding_cost":         total_holding_cost,
            "total_lost_sales":           total_lost_sales,
            "total_ordering_quantity":    total_ordering_quantity,
            "service_level":              service_level,
            "avg_inventory":              avg_inventory,
            "total_cost":                 total_cost,
            "n_steps":                    len(step_records),
        }

        logger.debug(
            f"KPIs: reward={total_cumulative_reward:.2f}, "
            f"lost_sales={total_lost_sales:.1f}, "
            f"service_level={service_level:.3f}"
        )

        return kpis

    def aggregate(self, episode_kpis: List[Dict]) -> Dict[str, float]:
        # Averages KPIs across multiple evaluation episodes
        if not episode_kpis:
            return self._empty_kpis()

        keys = [k for k in episode_kpis[0].keys() if k != "n_steps"]
        aggregated = {}

        for key in keys:
            values = np.array([e[key] for e in episode_kpis])
            aggregated[key]              = float(np.mean(values))
            aggregated[f"{key}_std"]     = float(np.std(values))

        aggregated["n_episodes"] = len(episode_kpis)
        return aggregated

    def _empty_kpis(self) -> Dict[str, float]:
        return {
            "total_cumulative_reward":  0.0,
            "total_holding_cost":       0.0,
            "total_lost_sales":         0.0,
            "total_ordering_quantity":  0.0,
            "service_level":            0.0,
            "avg_inventory":            0.0,
            "total_cost":               0.0,
            "n_steps":                  0,
        }
