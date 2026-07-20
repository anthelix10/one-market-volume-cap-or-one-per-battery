#!/usr/bin/env python3
"""Rebuild and validate cost and physical-comparability extensions.

This script implements two supporting analyses:

1. Re-optimised all-throughput costs for the per-unit-priced and shared-priced
   controllers. The cost is included in each optimisation objective rather than
   deducted from a schedule optimised at zero cost.
2. A physical-feasibility audit of the current PN declaration and B1610
   metering series against the stated asset power and energy capacities.

The script does not alter the retained core result files. It writes the
additional outputs listed in ``OUTPUTS`` to ``paper_of_record/``.

Examples
--------
  python code/run_cost_feasibility_extensions.py --rebuild-all
  python code/run_cost_feasibility_extensions.py --validate-stored
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unrecognized options detected")


def _load_core():
    path = Path(__file__).resolve().with_name("run_paper_of_record.py")
    spec = importlib.util.spec_from_file_location("bess_paper_of_record", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import core pipeline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = _load_core()

THROUGHPUT_FEES = [0.0, 2.5, 5.0, 10.0]
OUTPUTS = [
    "reoptimised_throughput_fee_daily.csv",
    "reoptimised_throughput_fee_summary.csv",
    "pn_b1610_physical_feasibility_daily_1081.csv",
    "pn_b1610_physical_feasibility_summary_1081.csv",
]


def _single_priced_with_fee(lp, forecast: np.ndarray, cap: np.ndarray, fee: float):
    n = core.PERIODS
    obj = np.zeros(lp.nvars)
    # scipy.linprog minimises. This is the negative of
    # forecast*(discharge-charge) - fee*(charge+discharge) - phi*slack.
    obj[:n] = core.DELTA * (forecast + fee)
    obj[n:2*n] = core.DELTA * (-forecast + fee)
    obj[2*n:4*n] = core.BASE_PHI * core.DELTA
    b = lp.base_b.copy()
    b[np.asarray(lp.cap_rows)] = np.repeat(cap, 2)
    bounds = [(0.0, lp.asset.pmax)] * (2*n) + [(0.0, None)] * (2*n)
    result = linprog(
        obj,
        A_ub=lp.Aub,
        b_ub=b,
        A_eq=lp.Aeq,
        b_eq=lp.beq,
        bounds=bounds,
        method="highs",
        options={"threads": 1},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[:n], result.x[n:2*n]


def _shared_priced_with_fee(lp, forecast: np.ndarray, cap: np.ndarray, fee: float):
    n = core.PERIODS
    n_assets = len(core.ASSETS)
    obj = np.zeros(lp.nvars)
    for asset_index in range(n_assets):
        obj[asset_index*n:(asset_index+1)*n] = core.DELTA * (forecast + fee)
        obj[lp.g0 + asset_index*n:lp.g0 + (asset_index+1)*n] = core.DELTA * (-forecast + fee)
    obj[lp.ec0:lp.ec0+n] = core.BASE_PHI * core.DELTA
    obj[lp.eg0:lp.eg0+n] = core.BASE_PHI * core.DELTA
    b = lp.base_b.copy()
    b[np.asarray(lp.cap_rows)] = np.repeat(cap, 2)
    result = linprog(
        obj,
        A_ub=lp.Aub,
        b_ub=b,
        A_eq=lp.Aeq,
        b_eq=lp.beq,
        bounds=lp.bounds,
        method="highs",
        options={"threads": 1},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return (
        result.x[:n_assets*n].reshape(n_assets, n),
        result.x[lp.g0:lp.g0+n_assets*n].reshape(n_assets, n),
    )


def _solve_fee_chunk(payload):
    """Solve a bounded fee/date chunk in a fresh worker process."""
    fee, start, date_values, fmat, pmat, vmat, svmat = payload
    single = {a.name: core.SingleLP(a, "priced") for a in core.ASSETS}
    shared = core.SharedLP("priced")
    rows = []
    for j, date_value in enumerate(date_values):
        date = pd.Timestamp(date_value)
        forecast = fmat[j]
        realised_price = pmat[j]
        realised_volume = vmat[j]
        scheduling_volume = svmat[j]

        per_c = np.zeros((len(core.ASSETS), core.PERIODS))
        per_g = np.zeros_like(per_c)
        for asset_index, asset in enumerate(core.ASSETS):
            cap = core.per_unit_cap(scheduling_volume, asset, core.BASE_RHO)
            per_c[asset_index], per_g[asset_index] = _single_priced_with_fee(
                single[asset.name], forecast, cap, fee
            )
        sha_c, sha_g = _shared_priced_with_fee(
            shared, forecast, core.shared_cap(scheduling_volume, core.BASE_RHO), fee
        )

        for label, c, g, audit_key in [
            ("Per-unit priced", per_c, per_g, "per_unit_score_gbp"),
            ("Shared priced", sha_c, sha_g, "shared_score_gbp"),
        ]:
            metrics = core.evaluate(
                c, g, realised_price, realised_volume, core.BASE_RHO, core.BASE_PHI
            )
            audit_score = float(metrics[audit_key])
            throughput = float(metrics["throughput_mwh"])
            rows.append({
                "date": date,
                "throughput_fee_gbp_mwh": fee,
                "schedule": label,
                "gross_gbp": float(metrics["gross_gbp"]),
                "audit_score_before_fee_gbp": audit_score,
                "throughput_mwh": throughput,
                "throughput_cost_gbp": fee * throughput,
                "net_conditional_score_gbp": audit_score - fee * throughput,
                "per_unit_excess_mwh": float(metrics["per_unit_excess_mwh"]),
                "shared_excess_mwh": float(metrics["shared_excess_mwh"]),
            })
    return start, rows


def rebuild_reoptimised_fees(root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import multiprocessing as mp

    reference = core.load_reference_readonly(root, out)
    dates = reference["matched"]
    forecasts = reference["base_forecast"].pivot(
        index="date", columns="period", values="forecast_price"
    )
    fmat = forecasts.loc[dates, range(1, 49)].to_numpy(float)
    pmat = reference["price"].loc[dates, core.PERIOD_COLS].to_numpy(float)
    vmat = reference["volume"].loc[dates, core.PERIOD_COLS].to_numpy(float)
    svmat = reference["sched_volume"].loc[dates, core.PERIOD_COLS].to_numpy(float)

    # Reuse the already validated zero-fee daily schedules.
    rows = []
    stored_daily = reference["daily"]
    for label, audit_key in [
        ("Per-unit priced", "per_unit_score_gbp"),
        ("Shared priced", "shared_score_gbp"),
    ]:
        subset = stored_daily[stored_daily["schedule"] == label]
        for _, r in subset.iterrows():
            throughput = float(r["throughput_mwh"])
            rows.append({
                "date": pd.Timestamp(r["date"]),
                "throughput_fee_gbp_mwh": 0.0,
                "schedule": label,
                "gross_gbp": float(r["gross_gbp"]),
                "audit_score_before_fee_gbp": float(r[audit_key]),
                "throughput_mwh": throughput,
                "throughput_cost_gbp": 0.0,
                "net_conditional_score_gbp": float(r[audit_key]),
                "per_unit_excess_mwh": float(r["per_unit_excess_mwh"]),
                "shared_excess_mwh": float(r["shared_excess_mwh"]),
            })

    # Use a small number of fresh bounded workers. Explicit single-threaded
    # HiGHS options avoid oversubscription, while one task per child prevents
    # the long-sequence slowdown observed after thousands of consecutive LPs.
    chunk = 180
    tasks = []
    for fee in [x for x in THROUGHPUT_FEES if x > 0]:
        for start in range(0, len(dates), chunk):
            end = min(start + chunk, len(dates))
            tasks.append((
                fee,
                start,
                dates[start:end].to_numpy(),
                fmat[start:end],
                pmat[start:end],
                vmat[start:end],
                svmat[start:end],
            ))
    context = mp.get_context("spawn" if sys.platform.startswith("win") else "fork")
    workers = min(3, mp.cpu_count() or 2)
    with context.Pool(processes=workers, maxtasksperchild=1) as pool:
        for completed, (_, part_rows) in enumerate(
            pool.imap_unordered(_solve_fee_chunk, tasks), start=1
        ):
            rows.extend(part_rows)
            print(f"reoptimised fee chunks {completed}/{len(tasks)}", flush=True)

    daily = pd.DataFrame(rows).sort_values(
        ["throughput_fee_gbp_mwh", "date", "schedule"]
    )
    summary = daily.groupby(
        ["throughput_fee_gbp_mwh", "schedule"], as_index=False
    ).agg(
        days=("date", "nunique"),
        gross_gbp=("gross_gbp", "sum"),
        audit_score_before_fee_gbp=("audit_score_before_fee_gbp", "sum"),
        throughput_mwh=("throughput_mwh", "sum"),
        throughput_mwh_day=("throughput_mwh", "mean"),
        throughput_cost_gbp=("throughput_cost_gbp", "sum"),
        net_conditional_score_gbp=("net_conditional_score_gbp", "sum"),
        per_unit_excess_mwh=("per_unit_excess_mwh", "sum"),
        shared_excess_mwh=("shared_excess_mwh", "sum"),
    )
    summary["gross_gbp_million"] = summary["gross_gbp"] / 1e6
    summary["audit_score_before_fee_gbp_million"] = summary["audit_score_before_fee_gbp"] / 1e6
    summary["net_conditional_score_gbp_million"] = summary["net_conditional_score_gbp"] / 1e6
    daily.to_csv(out / "reoptimised_throughput_fee_daily.csv", index=False)
    summary.to_csv(out / "reoptimised_throughput_fee_summary.csv", index=False)
    return daily, summary

def _physical_feasibility_for_series(
    df: pd.DataFrame,
    label: str,
    matched_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["settlementDate"].isin(matched_dates)].copy()
    if "quantity" in work.columns:
        work["charge_mwh"] = (-work["quantity"]).clip(lower=0.0)
        work["discharge_mwh"] = work["quantity"].clip(lower=0.0)
    else:
        work["charge_mwh"] = work["chargeMWh"]
        work["discharge_mwh"] = work["dischargeMWh"]

    daily_rows = []
    for asset in core.ASSETS:
        adf = work[work["asset"] == asset.name].copy()
        for date, day in adf.groupby("settlementDate"):
            day = day[day["settlementPeriod"].between(1, 48)].sort_values("settlementPeriod")
            if len(day) != core.PERIODS:
                raise RuntimeError(f"{label} incomplete on {date}: {asset.name}, {len(day)} rows")
            charge = day["charge_mwh"].fillna(0.0).to_numpy(float)
            discharge = day["discharge_mwh"].fillna(0.0).to_numpy(float)
            stored_movement = core.ETA_C * charge - discharge / core.ETA_D
            cumulative = np.r_[0.0, np.cumsum(stored_movement)]
            movement_range = float(cumulative.max() - cumulative.min())
            soc_from_half = asset.emax / 2.0 + cumulative
            max_charge_mw = float((charge / core.DELTA).max(initial=0.0))
            max_discharge_mw = float((discharge / core.DELTA).max(initial=0.0))
            daily_rows.append({
                "series": label,
                "asset": asset.name,
                "date": pd.Timestamp(date).date().isoformat(),
                "power_rating_mw": asset.pmax,
                "energy_capacity_mwh": asset.emax,
                "within_day_cumulative_range_mwh": movement_range,
                "feasible_for_some_initial_soc": bool(movement_range <= asset.emax + 1e-9),
                "feasible_from_50pct_soc": bool(
                    soc_from_half.min() >= -1e-9 and soc_from_half.max() <= asset.emax + 1e-9
                ),
                "minimum_cumulative_movement_mwh": float(cumulative.min()),
                "maximum_cumulative_movement_mwh": float(cumulative.max()),
                "gross_throughput_mwh": float(charge.sum() + discharge.sum()),
                "charge_mwh": float(charge.sum()),
                "discharge_mwh": float(discharge.sum()),
                "net_stored_movement_mwh": float(stored_movement.sum()),
                "maximum_charge_mw": max_charge_mw,
                "maximum_discharge_mw": max_discharge_mw,
                "power_rating_exceeded": bool(
                    max(max_charge_mw, max_discharge_mw) > asset.pmax + 1e-9
                ),
            })
    daily = pd.DataFrame(daily_rows)
    summary_rows = []
    for (series, asset), group in daily.groupby(["series", "asset"], sort=False):
        summary_rows.append({
            "series": series,
            "asset": asset,
            "days": int(len(group)),
            "feasible_for_some_initial_soc_days": int(group["feasible_for_some_initial_soc"].sum()),
            "feasible_for_some_initial_soc_percent": float(100.0 * group["feasible_for_some_initial_soc"].mean()),
            "feasible_from_50pct_soc_days": int(group["feasible_from_50pct_soc"].sum()),
            "feasible_from_50pct_soc_percent": float(100.0 * group["feasible_from_50pct_soc"].mean()),
            "power_rating_exceeded_days": int(group["power_rating_exceeded"].sum()),
            "maximum_within_day_range_mwh": float(group["within_day_cumulative_range_mwh"].max()),
            "maximum_daily_throughput_mwh": float(group["gross_throughput_mwh"].max()),
            "maximum_charge_mw_observed": float(group["maximum_charge_mw"].max()),
            "maximum_discharge_mw_observed": float(group["maximum_discharge_mw"].max()),
        })
    return daily, pd.DataFrame(summary_rows)


def rebuild_physical_feasibility(root: Path, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = core.load_reference_readonly(root, out)
    matched = reference["matched"]
    pn = pd.read_csv(
        root / "inputs/current_snapshot/current_latest_pn_asset_period_panel.csv",
        parse_dates=["settlementDate"],
    )
    b1610 = pd.read_csv(
        root / "inputs/current_snapshot/current_latest_b1610_asset_period_panel.csv",
        parse_dates=["settlementDate"],
    )
    pn_daily, pn_summary = _physical_feasibility_for_series(pn, "PN", matched)
    b_daily, b_summary = _physical_feasibility_for_series(b1610, "B1610", matched)
    daily = pd.concat([pn_daily, b_daily], ignore_index=True)
    summary = pd.concat([pn_summary, b_summary], ignore_index=True)
    daily.to_csv(out / "pn_b1610_physical_feasibility_daily_1081.csv", index=False)
    summary.to_csv(out / "pn_b1610_physical_feasibility_summary_1081.csv", index=False)
    return daily, summary


def validate_stored(root: Path, out: Path, report_path: Path | None = None) -> dict:
    missing = [name for name in OUTPUTS if not (out / name).exists()]
    if missing:
        raise RuntimeError(f"Missing final-extension outputs: {missing}")

    fee = pd.read_csv(out / "reoptimised_throughput_fee_summary.csv")
    phys = pd.read_csv(out / "pn_b1610_physical_feasibility_summary_1081.csv")
    reference = pd.read_csv(out / "controller_summary_1081.csv")

    # At zero fee, the re-optimised schedules must reproduce the stored reference totals.
    checks = []
    for schedule, metric in [
        ("Per-unit priced", "per_unit_score_gbp"),
        ("Shared priced", "shared_score_gbp"),
    ]:
        stored = float(reference.loc[reference["schedule"] == schedule, metric].iloc[0])
        rebuilt = float(fee.loc[
            (fee["schedule"] == schedule) & (fee["throughput_fee_gbp_mwh"] == 0.0),
            "net_conditional_score_gbp",
        ].iloc[0])
        checks.append({
            "check": f"zero_fee_{schedule.lower().replace(' ', '_')}",
            "stored_gbp": stored,
            "rebuilt_gbp": rebuilt,
            "absolute_difference_gbp": abs(stored - rebuilt),
            "pass": bool(abs(stored - rebuilt) <= 1e-5),
        })

    # Regression guards for the matched-panel physical-feasibility audit.
    expected = {
        ("PN", "Pillswood Battery Storage"): (1081, 748, 826.1752688172043),
        ("B1610", "Pillswood Battery Storage"): (1081, 1079, 197.87975268817206),
    }
    for key, (days, feasible_days, max_range) in expected.items():
        row = phys[(phys["series"] == key[0]) & (phys["asset"] == key[1])].iloc[0]
        ok = (
            int(row["days"]) == days
            and int(row["feasible_for_some_initial_soc_days"]) == feasible_days
            and abs(float(row["maximum_within_day_range_mwh"]) - max_range) <= 1e-3
        )
        checks.append({
            "check": f"physical_feasibility_{key[0].lower()}_{key[1].split()[0].lower()}",
            "expected_days": days,
            "actual_days": int(row["days"]),
            "expected_feasible_days": feasible_days,
            "actual_feasible_days": int(row["feasible_for_some_initial_soc_days"]),
            "expected_max_range_mwh": max_range,
            "actual_max_range_mwh": float(row["maximum_within_day_range_mwh"]),
            "pass": bool(ok),
        })

    report = {
        "status": "PASS" if all(x["pass"] for x in checks) else "FAIL",
        "checks": checks,
    }
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-all", action="store_true")
    parser.add_argument("--validate-stored", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional JSON validation-report path")
    args = parser.parse_args()
    if not (args.rebuild_all or args.validate_stored):
        parser.error("Choose --rebuild-all or --validate-stored")
    root = core.detect_root()
    out = core.output_root(root)
    started = time.time()
    if args.rebuild_all:
        rebuild_reoptimised_fees(root, out)
        rebuild_physical_feasibility(root, out)
    report = validate_stored(root, out, report_path=args.report)
    print(json.dumps({
        "status": report["status"],
        "runtime_seconds": time.time() - started,
        "outputs": OUTPUTS,
    }, indent=2))


if __name__ == "__main__":
    main()
