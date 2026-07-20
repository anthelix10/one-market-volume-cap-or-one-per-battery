#!/usr/bin/env python3
"""Generate the article figures and the essential supplementary figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POR = ROOT / "paper_of_record"
MAIN = ROOT / "generated_main_figures"
SUPP = ROOT / "supplementary_figures"
MAIN.mkdir(exist_ok=True)
SUPP.mkdir(exist_ok=True)

for folder in (MAIN, SUPP):
    for pattern in ("*.pdf", "*.png"):
        for path in folder.glob(pattern):
            path.unlink()

ASSETS = ["Pillswood", "Whitelee", "Roosecote"]
ASSET_FULL = ["Pillswood Battery Storage", "Whitelee 1 Battery", "Roosecote Battery"]
POWERS = np.array([98.0, 50.0, 49.0])
RHO = 0.02
DELTA = 0.5


def light_blues() -> LinearSegmentedColormap:
    """Return a light sequential colour map that keeps cell labels legible."""
    base = plt.get_cmap("Blues")
    return LinearSegmentedColormap.from_list(
        "light_blues", base(np.linspace(0.06, 0.70, 256))
    )


def save(fig: plt.Figure, folder: Path, stem: str) -> None:
    """Save vector and high-resolution raster copies."""
    fig.savefig(folder / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", suffix: str = "") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            fmt.format(value) + suffix,
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def surface_matrix(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        df.pivot(index="phi_gbp_mwh", columns="rho", values=column)
        .sort_index(ascending=False)
    )


def figure_1() -> None:
    dates = pd.DatetimeIndex(
        pd.read_csv(POR / "matched_dates_1081.csv", parse_dates=["date"])["date"]
    )
    cols = [str(i) for i in range(1, 49)]
    realised = (
        pd.read_csv(POR / "mid_volume_complete_2022_2025.csv", parse_dates=["date"])
        .set_index("date")
        .reindex(dates)[cols]
        .to_numpy(float)
        .ravel()
    )
    scheduling = (
        pd.read_csv(POR / "mid_volume_trailing28_median.csv", parse_dates=["date"])
        .set_index("date")
        .reindex(dates)[cols]
        .to_numpy(float)
        .ravel()
    )

    binding = []
    ratios: dict[str, np.ndarray] = {}
    for label, volume in [
        ("Realised audit volume", realised),
        ("Trailing-median scheduling volume", scheduling),
    ]:
        common = RHO * volume / DELTA
        binding.append(
            [100 * np.mean(common < power) for power in POWERS]
            + [100 * np.mean(common < POWERS.min())]
        )
        per_asset_total = np.minimum(POWERS[:, None], common[None, :]).sum(axis=0)
        aggregate = np.minimum(POWERS.sum(), common)
        mask = aggregate > 0
        ratios[label] = per_asset_total[mask] / aggregate[mask]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1))
    x = np.arange(4)
    width = 0.28
    bars_1 = axes[0].bar(
        x - width / 2, binding[0], width, label="Realised audit volume"
    )
    bars_2 = axes[0].bar(
        x + width / 2, binding[1], width, label="Trailing-median scheduling volume"
    )
    axes[0].set_xticks(x, ASSETS + ["All three"])
    axes[0].set_ylabel("Periods in which market-volume term binds (%)")
    axes[0].set_ylim(0, 86)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    annotate_bars(axes[0], bars_1, "{:.1f}", "%")
    annotate_bars(axes[0], bars_2, "{:.1f}", "%")
    axes[0].set_title("(a) Cap-binding frequency")

    bins = np.linspace(1.0, 3.05, 22)
    for label, ratio in ratios.items():
        axes[1].hist(
            ratio,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=label,
        )
    axes[1].axvline(1.0, linewidth=1, linestyle="--")
    axes[1].axvline(3.0, linewidth=1, linestyle=":")
    axes[1].set_xlabel("Sum of per-asset caps / one aggregate cap")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(0.95, 3.05)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.25)
    axes[1].set_title("(b) Implied aggregate-cap multiplier")
    axes[1].text(
        0.03,
        0.95,
        (
            f"Realised median: {np.median(ratios['Realised audit volume']):.2f}x\n"
            f"Scheduling median: {np.median(ratios['Trailing-median scheduling volume']):.2f}x"
        ),
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "alpha": 0.85,
            "edgecolor": "0.7",
        },
    )
    fig.tight_layout()
    save(fig, MAIN, "Figure_1_Cap_binding_and_aggregate_multiplier")


def figure_2() -> None:
    controller = pd.read_csv(POR / "controller_summary_1081.csv").set_index("schedule")
    counterfactual = pd.read_csv(
        POR / "aggregate_cap_counterfactual_summary.csv"
    ).set_index("variant")
    labels = [
        "Per-asset schedule\nper-asset audit",
        "Same per-asset schedule\naggregate audit",
        "Aggregate-optimised\naggregate 2% audit",
        "Aggregate-optimised\naggregate 6% audit",
        "Aggregate-optimised\nsum-of-asset caps",
    ]
    values = [
        controller.loc["Per-unit priced", "per_unit_score_gbp"] / 1e6,
        controller.loc["Per-unit priced", "shared_score_gbp"] / 1e6,
        counterfactual.loc["single_shared_2pct", "matching_audit_score_gbp"] / 1e6,
        counterfactual.loc[
            "shared_6pct_n_times_rho", "matching_audit_score_gbp"
        ]
        / 1e6,
        counterfactual.loc[
            "shared_sum_of_per_unit_caps", "matching_audit_score_gbp"
        ]
        / 1e6,
    ]
    fig, ax = plt.subplots(figsize=(9.6, 4.5))
    bars = ax.bar(np.arange(len(labels)), values, width=0.28)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Conditional score (GBP million)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars, "{:.2f}")

    accounting_gap = values[0] - values[1]
    recovery = values[2] - values[1]
    ax.annotate(
        "",
        xy=(0.5, values[0]),
        xytext=(0.5, values[1]),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0},
    )
    ax.text(
        0.5,
        values[0] + 0.65,
        f"Same-schedule accounting gap\nGBP {accounting_gap:.2f}m",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.annotate(
        "",
        xy=(1.5, values[2]),
        xytext=(1.5, values[1]),
        arrowprops={"arrowstyle": "<->", "linewidth": 1.0},
    )
    ax.text(
        1.5,
        values[2] + 0.65,
        f"Aggregate re-optimisation recovery\nGBP {recovery:.2f}m",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.set_title("Reference-case score decomposition and aggregate-cap counterfactuals")
    fig.tight_layout()
    save(fig, MAIN, "Figure_2_Reference_decomposition_and_counterfactuals")


def figure_3() -> None:
    df = pd.read_csv(POR / "cap_assignment_surface_25_cells.csv")
    metrics = [
        ("optimised_score_difference_gbp", "(a) Optimised per-asset minus aggregate score"),
        ("same_schedule_accounting_gap_gbp", "(b) Same-schedule accounting gap"),
        (
            "aggregate_reoptimisation_recovery_gbp",
            "(c) Recovery from aggregate re-optimisation",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.25), constrained_layout=True)
    all_values = np.concatenate([df[column].to_numpy(float) / 1e6 for column, _ in metrics])
    vmin, vmax = float(all_values.min()), float(all_values.max())
    image = None
    for ax, (column, title) in zip(axes, metrics):
        matrix = surface_matrix(df, column) / 1e6
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=light_blues(),
        )
        ax.set_xticks(range(len(matrix.columns)), [f"{100*x:g}%" for x in matrix.columns])
        ax.set_yticks(range(len(matrix.index)), [f"{int(x)}" for x in matrix.index])
        ax.set_xlabel("Participation fraction, rho")
        ax.set_ylabel("Penalty, phi (GBP/MWh)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix.iloc[row_idx, col_idx]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    colour_bar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colour_bar.set_label("GBP million")
    save(fig, MAIN, "Figure_3_Surface_decomposition")


def supplementary_figure_1() -> None:
    df = pd.read_csv(POR / "cap_assignment_surface_25_cells.csv")
    metrics = [
        ("per_unit_priced_per_unit_score_gbp", "Per-asset schedule / per-asset audit"),
        ("per_unit_priced_shared_score_gbp", "Same schedule / aggregate audit"),
        ("shared_priced_score_gbp", "Aggregate-optimised / aggregate audit"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    values = np.concatenate([df[column].to_numpy(float) / 1e6 for column, _ in metrics])
    image = None
    for ax, (column, title) in zip(axes, metrics):
        matrix = surface_matrix(df, column) / 1e6
        image = ax.imshow(
            matrix.to_numpy(),
            aspect="auto",
            vmin=values.min(),
            vmax=values.max(),
            cmap=light_blues(),
        )
        ax.set_xticks(range(len(matrix.columns)), [f"{100*x:g}%" for x in matrix.columns])
        ax.set_yticks(range(len(matrix.index)), [f"{int(x)}" for x in matrix.index])
        ax.set_xlabel("Participation fraction, rho")
        ax.set_ylabel("Penalty, phi (GBP/MWh)")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix.iloc[row_idx, col_idx]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
    colour_bar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
    colour_bar.set_label("GBP million")
    save(fig, SUPP, "Supplementary_Figure_S1_Matched_audit_score_matrices")


def supplementary_figure_2() -> None:
    grid = pd.read_csv(POR / "forecast_grid_current_20_cells.csv")
    matrix = (
        grid.pivot(index="training_days", columns="alpha", values="period_rmse_gbp_mwh")
        .sort_index(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=light_blues())
    ax.set_xticks(range(len(matrix.columns)), [f"{x:g}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [str(int(x)) for x in matrix.index])
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("Training window (days)")
    midpoint = float(np.median(matrix.to_numpy()))
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix.iloc[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > midpoint else "black",
            )
    colour_bar = fig.colorbar(image, ax=ax)
    colour_bar.set_label("Period RMSE (GBP/MWh)")
    fig.tight_layout()
    save(fig, SUPP, "Supplementary_Figure_S2_Ridge_forecast_grid")


def supplementary_figure_3() -> None:
    daily = pd.read_csv(POR / "pn_b1610_physical_feasibility_daily_1081.csv")
    daily["range_capacity_ratio"] = (
        daily["within_day_cumulative_range_mwh"] / daily["energy_capacity_mwh"]
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), sharey=True)
    for ax, asset, short_name in zip(axes, ASSET_FULL, ASSETS):
        subset = daily[daily.asset == asset]
        upper = max(4.5, float(subset.range_capacity_ratio.max()))
        for series, group in subset.groupby("series"):
            ax.hist(
                group["range_capacity_ratio"],
                bins=np.linspace(0, upper, 40),
                density=True,
                histtype="step",
                label=series,
            )
        ax.axvline(1.0, linestyle="--", linewidth=1)
        ax.set_title(short_name)
        ax.set_xlabel("Within-day range / capacity")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Density")
    axes[-1].legend(frameon=False)
    fig.tight_layout()
    save(fig, SUPP, "Supplementary_Figure_S3_Physical_feasibility_distributions")


def main() -> None:
    figure_1()
    figure_2()
    figure_3()
    supplementary_figure_1()
    supplementary_figure_2()
    supplementary_figure_3()
    print("Generated three article figures and three supplementary figures")


if __name__ == "__main__":
    main()
