import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

logger = logging.getLogger(__name__)

# Consistent colours for each agent across all plots
AGENT_COLORS = {
    "RL+LSTM Agent":       "#1565C0",
    "Standard RL Agent":   "#2E7D32",
    "Fixed-S Policy":      "#E65100",
    "Forecast Base-Stock": "#6A1B9A",
}


def plot_kpi_comparison(
    df:        pd.DataFrame,
    save_dir:  str  = "outputs/plots/",
    show:      bool = False,
) -> None:
    # Generates grouped by bar charts comparing all 4 agents on each KPI, faceted by scenario
    os.makedirs(save_dir, exist_ok=True)

    kpi_configs = [
        ("Cumulative Reward",   "Total Cumulative Reward",         True),
        ("Holding Cost",        "Total Inventory Holding Cost",     False),
        ("Lost Sales (units)",  "Total Lost Sales (Units)",         False),
        ("Ordering Quantity",   "Total Ordering Quantity (Units)",  False),
        ("Service Level (%)",   "Service Level (%)",                True),
        ("Total Cost",          "Total Operational Cost",           False),
    ]

    scenarios = df["Scenario"].unique()
    agents    = list(AGENT_COLORS.keys())
    n_agents  = len(agents)

    for col, title, higher_is_better in kpi_configs:
        if col not in df.columns:
            continue

        fig, axes = plt.subplots(
            1, len(scenarios),
            figsize=(5 * len(scenarios), 6),
            sharey=False,
        )
        if len(scenarios) == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=14, fontweight="bold")

        for ax, scenario in zip(axes, scenarios):
            scenario_df = df[df["Scenario"] == scenario]
            avg_by_agent = scenario_df.groupby("Agent")[col].mean()

            x      = np.arange(n_agents)
            values = [avg_by_agent.get(a, 0) for a in agents]
            colors = [AGENT_COLORS.get(a, "gray") for a in agents]

            bars = ax.bar(x, values, color=colors, alpha=0.85, width=0.6)

            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8
                )

            ax.set_title(scenario)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [a.replace(" ", "\n") for a in agents],
                fontsize=8
            )
            ax.set_ylabel(col if scenario == scenarios[0] else "")
            ax.grid(True, axis="y", alpha=0.3)

            # Highlight best agent
            best_idx = int(np.argmax(values) if higher_is_better else np.argmin(values))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(2.5)

        plt.tight_layout()
        filename = col.lower().replace(" ", "_").replace("(%)", "pct").replace("(units)", "") + ".png"
        save_path = os.path.join(save_dir, filename)
        _save_or_show(fig, save_path, show)
        logger.info(f"Saved: {save_path}")


def plot_cost_breakdown(
    df:       pd.DataFrame,
    scenario: str  = "Scenario A",
    save_dir: str  = "outputs/plots/",
    show:     bool = False,
) -> None:
    # Builds stacked bar chart showing cost breakdown per agent for one scenario
    os.makedirs(save_dir, exist_ok=True)

    scenario_df = df[df["Scenario"] == scenario]
    agents      = list(AGENT_COLORS.keys())

    holding  = [scenario_df[scenario_df["Agent"] == a]["Holding Cost"].mean()        for a in agents]
    lost     = [scenario_df[scenario_df["Agent"] == a]["Lost Sales (units)"].mean()   for a in agents]
    ordering = [scenario_df[scenario_df["Agent"] == a]["Ordering Quantity"].mean()    for a in agents]

    x   = np.arange(len(agents))
    w   = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    b1 = ax.bar(x, holding,  w, label="Holding Cost",    color="#42A5F5", alpha=0.85)
    b2 = ax.bar(x, lost,     w, bottom=holding,          label="Lost Sales Cost", color="#EF5350", alpha=0.85)
    b3 = ax.bar(x, ordering, w, bottom=np.array(holding)+np.array(lost),
                label="Ordering Cost", color="#66BB6A", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([a.replace(" ", "\n") for a in agents], fontsize=9)
    ax.set_ylabel("Cost (units)")
    ax.set_title(f"Cost Breakdown by Agent — {scenario}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"cost_breakdown_{scenario.replace(' ', '_').lower()}.png")
    _save_or_show(fig, save_path, show)


def plot_service_level_comparison(
    df:       pd.DataFrame,
    save_dir: str  = "outputs/plots/",
    show:     bool = False,
) -> None:
    # Line chart showing service level per agent across all 3 scenarios
    os.makedirs(save_dir, exist_ok=True)

    scenarios = sorted(df["Scenario"].unique())
    agents    = list(AGENT_COLORS.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    for agent in agents:
        agent_df = df[df["Agent"] == agent]
        values   = [agent_df[agent_df["Scenario"] == s]["Service Level (%)"].mean()
                    for s in scenarios]
        ax.plot(
            scenarios, values,
            marker="o", linewidth=2, label=agent,
            color=AGENT_COLORS.get(agent, "gray")
        )

    ax.set_ylabel("Service Level (%)")
    ax.set_title("Service Level Across Scenarios — All Agents")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "service_level_comparison.png")
    _save_or_show(fig, save_path, show)


def _save_or_show(fig, save_path, show):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)