#!/usr/bin/env python3
"""Rebuild and validate cap-assignment and total-cap counterfactuals.

The retained scope analysis contains two parts: the 25-cell per-asset schedule
surface evaluated under both audit conventions, and two aggregate-controller
counterfactuals that preserve more of the notional total participation budget.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_paper_of_record as base

GZIP = {"method": "gzip", "compresslevel": 6, "mtime": 0}
_WORKER = {}


def _init_surface_worker(dates, forecast, realised_price, realised_volume, scheduling_volume):
    global _WORKER
    _WORKER = {
        "dates": dates,
        "forecast": forecast,
        "price": realised_price,
        "volume": realised_volume,
        "scheduling_volume": scheduling_volume,
    }


def _surface_worker(task):
    rho, start, end = task
    solvers = {asset.name: base.SingleLP(asset, "priced") for asset in base.ASSETS}
    rows = []
    for index in range(start, end):
        date = pd.Timestamp(_WORKER["dates"][index])
        forecast = _WORKER["forecast"][index]
        realised_price = _WORKER["price"][index]
        realised_volume = _WORKER["volume"][index]
        scheduling_volume = _WORKER["scheduling_volume"][index]
        for phi in base.PHI_GRID:
            charge = np.zeros((len(base.ASSETS), base.PERIODS))
            discharge = np.zeros_like(charge)
            for asset_index, asset in enumerate(base.ASSETS):
                cap = base.per_unit_cap(scheduling_volume, asset, rho)
                charge[asset_index], discharge[asset_index] = solvers[asset.name].solve(
                    forecast, cap, phi
                )
            metric = base.evaluate(
                charge, discharge, realised_price, realised_volume, rho, phi
            )
            rows.append({
                "date": date,
                "rho": rho,
                "phi_gbp_mwh": phi,
                "per_unit_priced_gross_gbp": metric["gross_gbp"],
                "per_unit_priced_per_unit_score_gbp": metric["per_unit_score_gbp"],
                "per_unit_priced_shared_score_gbp": metric["shared_score_gbp"],
                "per_unit_priced_throughput_mwh": metric["throughput_mwh"],
                "per_unit_priced_per_unit_excess_mwh": metric["per_unit_excess_mwh"],
                "per_unit_priced_shared_excess_mwh": metric["shared_excess_mwh"],
            })
    return rows


def rebuild_surface(root: Path, out: Path, workers: int | None = None):
    reference = base.load_reference_readonly(root, out)
    dates = reference["matched"]
    forecast = reference["base_forecast"].pivot(
        index="date", columns="period", values="forecast_price"
    ).loc[dates, range(1, 49)].to_numpy(float)
    realised_price = reference["price"].loc[dates, base.PERIOD_COLS].to_numpy(float)
    realised_volume = reference["volume"].loc[dates, base.PERIOD_COLS].to_numpy(float)
    scheduling_volume = reference["sched_volume"].loc[dates, base.PERIOD_COLS].to_numpy(float)

    chunk = 35
    tasks = [
        (rho, start, min(start + chunk, len(dates)))
        for rho in base.RHO_GRID
        for start in range(0, len(dates), chunk)
    ]
    process_count = workers or min(12, os.cpu_count() or 2)
    context = mp.get_context("spawn" if os.name == "nt" else "fork")
    rows = []
    with context.Pool(
        process_count,
        initializer=_init_surface_worker,
        initargs=(dates.to_numpy(), forecast, realised_price, realised_volume, scheduling_volume),
        maxtasksperchild=2,
    ) as pool:
        for index, part in enumerate(pool.imap_unordered(_surface_worker, tasks), 1):
            rows.extend(part)
            if index % 10 == 0 or index == len(tasks):
                print(f"per-asset surface chunks {index}/{len(tasks)}", flush=True)

    daily = pd.DataFrame(rows).sort_values(["rho", "phi_gbp_mwh", "date"])
    daily.to_csv(
        out / "cap_assignment_surface_daily.csv.gz",
        index=False,
        compression=GZIP,
    )
    summary = daily.groupby(["rho", "phi_gbp_mwh"], as_index=False).agg(
        per_unit_priced_gross_gbp=("per_unit_priced_gross_gbp", "sum"),
        per_unit_priced_per_unit_score_gbp=("per_unit_priced_per_unit_score_gbp", "sum"),
        per_unit_priced_shared_score_gbp=("per_unit_priced_shared_score_gbp", "sum"),
        per_unit_priced_throughput_mwh_day=("per_unit_priced_throughput_mwh", "mean"),
        per_unit_priced_per_unit_excess_mwh_day=("per_unit_priced_per_unit_excess_mwh", "mean"),
        per_unit_priced_shared_excess_mwh_day=("per_unit_priced_shared_excess_mwh", "mean"),
    )

    shared_daily = pd.read_csv(out / "parameter_surface_daily_results.csv.gz")
    shared = shared_daily.groupby(
        ["rho", "phi_gbp_mwh", "schedule"], as_index=False
    ).shared_score_gbp.sum().pivot(
        index=["rho", "phi_gbp_mwh"], columns="schedule", values="shared_score_gbp"
    ).reset_index().rename(columns={
        "PN declaration": "pn_score_gbp",
        "Shared hard": "shared_hard_score_gbp",
        "Shared priced": "shared_priced_score_gbp",
    })
    summary = summary.merge(
        shared[[
            "rho", "phi_gbp_mwh", "pn_score_gbp",
            "shared_hard_score_gbp", "shared_priced_score_gbp",
        ]],
        on=["rho", "phi_gbp_mwh"],
        validate="one_to_one",
    )
    summary["optimised_score_difference_gbp"] = (
        summary["per_unit_priced_per_unit_score_gbp"] - summary["shared_priced_score_gbp"]
    )
    summary["same_schedule_accounting_gap_gbp"] = (
        summary["per_unit_priced_per_unit_score_gbp"]
        - summary["per_unit_priced_shared_score_gbp"]
    )
    summary["aggregate_reoptimisation_recovery_gbp"] = (
        summary["shared_priced_score_gbp"]
        - summary["per_unit_priced_shared_score_gbp"]
    )
    summary["optimised_score_difference_positive"] = (
        summary["optimised_score_difference_gbp"] > 0
    )
    summary.to_csv(out / "cap_assignment_surface_25_cells.csv", index=False)
    return daily, summary


def _evaluate_with_cap(charge, discharge, price, audit_cap, phi):
    gross = float(np.sum(base.DELTA * price[None, :] * (discharge - charge)))
    throughput = float(np.sum(base.DELTA * (discharge + charge)))
    excess = float(np.sum(base.DELTA * np.maximum(charge.sum(axis=0) - audit_cap, 0.0)))
    excess += float(np.sum(base.DELTA * np.maximum(discharge.sum(axis=0) - audit_cap, 0.0)))
    return gross, gross - phi * excess, throughput, excess


def rebuild_allowance_counterfactuals(root: Path, out: Path):
    reference = base.load_reference_readonly(root, out)
    dates = reference["matched"]
    forecast_map = base.forecast_map(reference["base_forecast"])
    solver = base.SharedLP("priced")
    stored = reference["daily"]
    stored = stored[stored.schedule == "Shared priced"].copy()
    rows = []
    for row in stored.itertuples(index=False):
        rows.append({
            "date": row.date,
            "variant": "single_shared_2pct",
            "gross_gbp": float(row.gross_gbp),
            "matching_audit_score_gbp": float(row.shared_score_gbp),
            "per_unit_audit_score_gbp": float(row.per_unit_score_gbp),
            "throughput_mwh": float(row.throughput_mwh),
            "matching_excess_mwh": float(row.shared_excess_mwh),
        })
    for date in dates:
        forecast = forecast_map[date]
        realised_price = reference["price"].loc[date, base.PERIOD_COLS].to_numpy(float)
        realised_volume = reference["volume"].loc[date, base.PERIOD_COLS].to_numpy(float)
        scheduling_volume = reference["sched_volume"].loc[date, base.PERIOD_COLS].to_numpy(float)
        variants = {
            "shared_6pct_n_times_rho": (
                base.shared_cap(scheduling_volume, 3 * base.BASE_RHO),
                base.shared_cap(realised_volume, 3 * base.BASE_RHO),
            ),
            "shared_sum_of_per_unit_caps": (
                sum(base.per_unit_cap(scheduling_volume, asset, base.BASE_RHO) for asset in base.ASSETS),
                sum(base.per_unit_cap(realised_volume, asset, base.BASE_RHO) for asset in base.ASSETS),
            ),
        }
        for name, (scheduling_cap, audit_cap) in variants.items():
            charge, discharge = solver.solve(forecast, scheduling_cap, base.BASE_PHI)
            gross, matching_score, throughput, matching_excess = _evaluate_with_cap(
                charge, discharge, realised_price, audit_cap, base.BASE_PHI
            )
            per_unit_score = base.evaluate(
                charge,
                discharge,
                realised_price,
                realised_volume,
                base.BASE_RHO,
                base.BASE_PHI,
            )["per_unit_score_gbp"]
            rows.append({
                "date": date,
                "variant": name,
                "gross_gbp": gross,
                "matching_audit_score_gbp": matching_score,
                "per_unit_audit_score_gbp": per_unit_score,
                "throughput_mwh": throughput,
                "matching_excess_mwh": matching_excess,
            })
    daily = pd.DataFrame(rows).sort_values(["variant", "date"])
    daily.to_csv(
        out / "aggregate_cap_counterfactual_daily.csv.gz",
        index=False,
        compression=GZIP,
    )
    summary = daily.groupby("variant", as_index=False).agg(
        days=("date", "nunique"),
        gross_gbp=("gross_gbp", "sum"),
        matching_audit_score_gbp=("matching_audit_score_gbp", "sum"),
        per_unit_audit_score_gbp=("per_unit_audit_score_gbp", "sum"),
        throughput_mwh_day=("throughput_mwh", "mean"),
        matching_excess_mwh_day=("matching_excess_mwh", "mean"),
    )
    summary.to_csv(out / "aggregate_cap_counterfactual_summary.csv", index=False)
    return daily, summary


def validate_stored(root: Path, out: Path, tol: float = 1e-6):
    required = [
        "cap_assignment_surface_daily.csv.gz",
        "cap_assignment_surface_25_cells.csv",
        "parameter_surface_daily_results.csv.gz",
        "aggregate_cap_counterfactual_daily.csv.gz",
        "aggregate_cap_counterfactual_summary.csv",
        "controller_summary_1081.csv",
    ]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"Missing scope-extension outputs: {missing}")

    daily = pd.read_csv(out / "cap_assignment_surface_daily.csv.gz")
    stored = pd.read_csv(out / "cap_assignment_surface_25_cells.csv")
    calculated = daily.groupby(["rho", "phi_gbp_mwh"], as_index=False).agg(
        per_unit_priced_gross_gbp=("per_unit_priced_gross_gbp", "sum"),
        per_unit_priced_per_unit_score_gbp=("per_unit_priced_per_unit_score_gbp", "sum"),
        per_unit_priced_shared_score_gbp=("per_unit_priced_shared_score_gbp", "sum"),
    )
    merged = calculated.merge(stored, on=["rho", "phi_gbp_mwh"], suffixes=("_calc", "_stored"))
    fields = [
        "per_unit_priced_gross_gbp",
        "per_unit_priced_per_unit_score_gbp",
        "per_unit_priced_shared_score_gbp",
    ]
    surface_error = max(
        float(np.max(np.abs(merged[field + "_calc"] - merged[field + "_stored"])))
        for field in fields
    )

    shared_daily = pd.read_csv(out / "parameter_surface_daily_results.csv.gz")
    shared = shared_daily.groupby(
        ["rho", "phi_gbp_mwh", "schedule"], as_index=False
    ).shared_score_gbp.sum().pivot(
        index=["rho", "phi_gbp_mwh"], columns="schedule", values="shared_score_gbp"
    ).reset_index().rename(columns={
        "PN declaration": "pn_score_gbp",
        "Shared hard": "shared_hard_score_gbp",
        "Shared priced": "shared_priced_score_gbp",
    })
    shared_merge = shared.merge(stored, on=["rho", "phi_gbp_mwh"], suffixes=("_calc", "_stored"))
    shared_error = max(
        float(np.max(np.abs(shared_merge[field + "_calc"] - shared_merge[field + "_stored"])))
        for field in ["pn_score_gbp", "shared_hard_score_gbp", "shared_priced_score_gbp"]
    )

    positive_cells = int(stored["optimised_score_difference_positive"].sum())
    minimum_difference = float(stored["optimised_score_difference_gbp"].min())
    maximum_difference = float(stored["optimised_score_difference_gbp"].max())

    allowance_daily = pd.read_csv(out / "aggregate_cap_counterfactual_daily.csv.gz")
    allowance_stored = pd.read_csv(out / "aggregate_cap_counterfactual_summary.csv")
    allowance_calculated = allowance_daily.groupby("variant", as_index=False).agg(
        gross_gbp=("gross_gbp", "sum"),
        matching_audit_score_gbp=("matching_audit_score_gbp", "sum"),
        per_unit_audit_score_gbp=("per_unit_audit_score_gbp", "sum"),
    )
    allowance_merge = allowance_calculated.merge(
        allowance_stored, on="variant", suffixes=("_calc", "_stored")
    )
    allowance_error = max(
        float(np.max(np.abs(allowance_merge[field + "_calc"] - allowance_merge[field + "_stored"])))
        for field in ["gross_gbp", "matching_audit_score_gbp", "per_unit_audit_score_gbp"]
    )

    reference = pd.read_csv(out / "controller_summary_1081.csv")
    shared_priced = reference.loc[reference.schedule == "Shared priced"].iloc[0]
    baseline = allowance_stored.loc[allowance_stored.variant == "single_shared_2pct"].iloc[0]
    reference_error = max(
        abs(float(baseline.gross_gbp) - float(shared_priced.gross_gbp)),
        abs(float(baseline.matching_audit_score_gbp) - float(shared_priced.shared_score_gbp)),
        abs(float(baseline.per_unit_audit_score_gbp) - float(shared_priced.per_unit_score_gbp)),
    )

    max_error = max(surface_error, shared_error, allowance_error, reference_error)
    status = "PASS" if max_error <= tol and positive_cells == 25 else "FAIL"
    report = {
        "status": status,
        "tolerance": tol,
        "surface_daily_to_summary_max_abs_gbp": surface_error,
        "shared_surface_to_cap_summary_max_abs_gbp": shared_error,
        "allowance_daily_to_summary_max_abs_gbp": allowance_error,
        "allowance_baseline_to_reference_max_abs_gbp": reference_error,
        "positive_optimised_score_difference_cells": positive_cells,
        "minimum_optimised_score_difference_gbp": minimum_difference,
        "maximum_optimised_score_difference_gbp": maximum_difference,
        "shared_6pct_matching_score_gbp": float(
            allowance_stored.loc[
                allowance_stored.variant == "shared_6pct_n_times_rho",
                "matching_audit_score_gbp",
            ].iloc[0]
        ),
        "sum_of_caps_matching_score_gbp": float(
            allowance_stored.loc[
                allowance_stored.variant == "shared_sum_of_per_unit_caps",
                "matching_audit_score_gbp",
            ].iloc[0]
        ),
    }
    if status != "PASS":
        raise RuntimeError(report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--rebuild-surface", action="store_true")
    parser.add_argument("--rebuild-allowance", action="store_true")
    parser.add_argument("--validate-stored", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    root = base.detect_root()
    out = base.output_root(root)
    if args.rebuild or args.rebuild_surface:
        started = time.time()
        rebuild_surface(root, out, workers=args.workers)
        if args.rebuild:
            rebuild_allowance_counterfactuals(root, out)
        print(f"scope outputs rebuilt in {time.time() - started:.1f} s")
    if args.rebuild_allowance:
        rebuild_allowance_counterfactuals(root, out)
    if args.validate_stored:
        validate_stored(root, out)
    if not any(vars(args).values()):
        parser.print_help()


if __name__ == "__main__":
    main()
