#!/usr/bin/env python3
"""Evaluate all economic-unit partitions of the three-asset portfolio.

The article formalises cap assignment as a partition of physical assets into
economic units. This script computes the all-singleton and one-block endpoints
and the three intermediate two-block partitions at the reference rho and phi.
It reports a fixed-schedule audit of the retained per-asset schedule and
reoptimised schedules under each partition.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

DELTA = 0.5
ETA_C = 0.93
ETA_D = 0.93
RHO = 0.02
PHI = 30.0
PERIODS = 48
PERIOD_COLS = [str(i) for i in range(1, 49)]

@dataclass(frozen=True)
class Asset:
    name: str
    pmax: float
    emax: float

ASSETS = [
    Asset("Pillswood Battery Storage", 98.0, 196.0),
    Asset("Whitelee 1 Battery", 50.0, 50.0),
    Asset("Roosecote Battery", 49.0, 24.5),
]

PARTITIONS: dict[str, tuple[tuple[int, ...], ...]] = {
    "Pillswood | Whitelee | Roosecote": ((0,), (1,), (2,)),
    "Pillswood+Whitelee | Roosecote": ((0, 1), (2,)),
    "Pillswood+Roosecote | Whitelee": ((0, 2), (1,)),
    "Whitelee+Roosecote | Pillswood": ((1, 2), (0,)),
    "Pillswood+Whitelee+Roosecote": ((0, 1, 2),),
}
INTERMEDIATE = {k: v for k, v in PARTITIONS.items() if len(v) == 2}

# Fork-shared arrays set in rebuild().
_G: dict[str, object] = {}


def detect_root() -> Path:
    here = Path(__file__).resolve()
    if (here.parents[1] / "inputs" / "current_snapshot").exists():
        return here.parents[1]
    env = os.environ.get("BESS_SUPPLEMENT_ROOT")
    if env:
        return Path(env)
    raise RuntimeError("Could not locate package root")


def output_root(root: Path) -> Path:
    p = Path(os.environ.get("BESS_OUTPUT_ROOT", root / "paper_of_record"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def group_cap(volume: np.ndarray, group: tuple[int, ...], rho: float = RHO) -> np.ndarray:
    return np.minimum(sum(ASSETS[i].pmax for i in group), rho * volume / DELTA)


def audit_partition(c, g, price, volume, partition, rho=RHO, phi=PHI):
    gross = float(np.sum(DELTA * price[None, :] * (g - c)))
    charge_excess = 0.0
    discharge_excess = 0.0
    for group in partition:
        cap = group_cap(volume, group, rho)
        gc = c[list(group)].sum(axis=0)
        gg = g[list(group)].sum(axis=0)
        charge_excess += float(np.sum(DELTA * np.maximum(gc - cap, 0.0)))
        discharge_excess += float(np.sum(DELTA * np.maximum(gg - cap, 0.0)))
    excess = charge_excess + discharge_excess
    return {
        "gross_gbp": gross,
        "excess_mwh": excess,
        "score_gbp": gross - phi * excess,
        "throughput_mwh": float(np.sum(DELTA * (c + g))),
    }


class PartitionLP:
    def __init__(self, partition):
        self.partition = partition
        A, G, n = len(ASSETS), len(partition), PERIODS
        self.g0 = A * n
        self.ec0 = 2 * A * n
        self.eg0 = self.ec0 + G * n
        self.nvars = 2 * A * n + 2 * G * n
        rows, cols, data, b = [], [], [], []
        rowno = 0
        for asset_idx, asset in enumerate(ASSETS):
            for t in range(1, n + 1):
                for sign, rhs in ((1.0, asset.emax / 2.0), (-1.0, asset.emax / 2.0)):
                    for k in range(t):
                        rows.extend((rowno, rowno))
                        cols.extend((asset_idx * n + k, self.g0 + asset_idx * n + k))
                        data.extend((sign * ETA_C * DELTA, sign * (-DELTA / ETA_D)))
                    b.append(rhs); rowno += 1
        self.cap_rows = []
        for gi, group in enumerate(partition):
            for t in range(n):
                for side in (0, 1):
                    for asset_idx in group:
                        idx = asset_idx * n + t if side == 0 else self.g0 + asset_idx * n + t
                        rows.append(rowno); cols.append(idx); data.append(1.0)
                    slack = self.ec0 + gi * n + t if side == 0 else self.eg0 + gi * n + t
                    rows.append(rowno); cols.append(slack); data.append(-1.0)
                    b.append(0.0); self.cap_rows.append(rowno); rowno += 1
        self.Aub = sparse.csr_matrix((data, (rows, cols)), shape=(rowno, self.nvars))
        self.base_b = np.asarray(b, float)
        er, ec, ed = [], [], []
        for asset_idx in range(A):
            for t in range(n):
                er.extend((asset_idx, asset_idx))
                ec.extend((asset_idx * n + t, self.g0 + asset_idx * n + t))
                ed.extend((ETA_C * DELTA, -DELTA / ETA_D))
        self.Aeq = sparse.csr_matrix((ed, (er, ec)), shape=(A, self.nvars))
        self.beq = np.zeros(A)
        self.bounds = []
        for asset in ASSETS: self.bounds += [(0.0, asset.pmax)] * n
        for asset in ASSETS: self.bounds += [(0.0, asset.pmax)] * n
        self.bounds += [(0.0, None)] * (2 * G * n)

    def solve(self, forecast, volume):
        A, G, n = len(ASSETS), len(self.partition), PERIODS
        obj = np.zeros(self.nvars)
        for asset_idx in range(A):
            obj[asset_idx*n:(asset_idx+1)*n] = DELTA * forecast
            obj[self.g0+asset_idx*n:self.g0+(asset_idx+1)*n] = -DELTA * forecast
        obj[self.ec0:self.eg0+G*n] = PHI * DELTA
        b = self.base_b.copy()
        pos = 0
        for group in self.partition:
            cap = group_cap(volume, group)
            for t in range(n):
                b[self.cap_rows[pos]] = cap[t]; pos += 1
                b[self.cap_rows[pos]] = cap[t]; pos += 1
        res = linprog(obj, A_ub=self.Aub, b_ub=b, A_eq=self.Aeq, b_eq=self.beq,
                      bounds=self.bounds, method="highs", options={"presolve": True})
        if not res.success:
            raise RuntimeError(res.message)
        c = res.x[:A*n].reshape(A, n)
        g = res.x[self.g0:self.g0+A*n].reshape(A, n)
        return c, g


def _worker(indices):
    solvers = {name: PartitionLP(partition) for name, partition in INTERMEDIATE.items()}
    rows = []
    for i in indices:
        date = _G["dates"][i]
        price = _G["price"][i]
        volume = _G["volume"][i]
        sched_volume = _G["sched_volume"][i]
        forecast = _G["forecast"][i]
        fixed_c = _G["fixed_c"][i]
        fixed_g = _G["fixed_g"][i]
        for name, partition in INTERMEDIATE.items():
            fixed = audit_partition(fixed_c, fixed_g, price, volume, partition)
            c, g = solvers[name].solve(forecast, sched_volume)
            opt = audit_partition(c, g, price, volume, partition)
            rows.append({
                "date": date,
                "partition": name,
                "groups": 2,
                "fixed_per_asset_schedule_score_gbp": fixed["score_gbp"],
                "fixed_per_asset_schedule_gross_gbp": fixed["gross_gbp"],
                "fixed_per_asset_schedule_excess_mwh": fixed["excess_mwh"],
                "optimised_score_gbp": opt["score_gbp"],
                "optimised_gross_gbp": opt["gross_gbp"],
                "optimised_excess_mwh": opt["excess_mwh"],
                "optimised_throughput_mwh": opt["throughput_mwh"],
            })
    return rows


def _load_parent_arrays(out: Path):
    price_df = pd.read_csv(out / "mid_price_complete_2022_2025.csv", parse_dates=["date"]).set_index("date")
    volume_df = pd.read_csv(out / "mid_volume_complete_2022_2025.csv", parse_dates=["date"]).set_index("date")
    sched_df = pd.read_csv(out / "mid_volume_trailing28_median.csv", parse_dates=["date"]).set_index("date")
    forecast_df = pd.read_csv(out / "forecast_base_ridge_period.csv", parse_dates=["date"])
    period = pd.read_csv(out / "controller_period_schedules_1081.csv.gz", parse_dates=["date"])
    dates = pd.DatetimeIndex(sorted(period.date.unique()))
    fmap = {pd.Timestamp(d): g.sort_values("period").forecast_price.to_numpy(float) for d, g in forecast_df.groupby("date")}
    price = np.stack([price_df.loc[d, PERIOD_COLS].to_numpy(float) for d in dates])
    volume = np.stack([volume_df.loc[d, PERIOD_COLS].to_numpy(float) for d in dates])
    sched = np.stack([sched_df.loc[d, PERIOD_COLS].to_numpy(float) for d in dates])
    forecast = np.stack([fmap[d] for d in dates])
    fixed_c = np.zeros((len(dates), len(ASSETS), PERIODS))
    fixed_g = np.zeros_like(fixed_c)
    sub = period[period.schedule == "Per-unit priced"].copy()
    for asset_idx, asset in enumerate(ASSETS):
        a = sub[sub.asset == asset.name].sort_values(["date", "settlement_period"])
        fixed_c[:, asset_idx, :] = a.charge_mw.to_numpy(float).reshape(len(dates), PERIODS)
        fixed_g[:, asset_idx, :] = a.discharge_mw.to_numpy(float).reshape(len(dates), PERIODS)
    return dates, price, volume, sched, forecast, fixed_c, fixed_g


def _endpoint_rows(out: Path, dates: pd.DatetimeIndex):
    daily = pd.read_csv(out / "controller_daily_results_1081.csv", parse_dates=["date"])
    pu = daily[daily.schedule == "Per-unit priced"].sort_values("date")
    sh = daily[daily.schedule == "Shared priced"].sort_values("date")
    if not (pu.date.to_numpy() == dates.to_numpy()).all() or not (sh.date.to_numpy() == dates.to_numpy()).all():
        raise RuntimeError("Stored daily dates do not match")
    rows = []
    for r in pu.itertuples(index=False):
        rows.append({
            "date": r.date,
            "partition": "Pillswood | Whitelee | Roosecote",
            "groups": 3,
            "fixed_per_asset_schedule_score_gbp": r.per_unit_score_gbp,
            "fixed_per_asset_schedule_gross_gbp": r.gross_gbp,
            "fixed_per_asset_schedule_excess_mwh": r.per_unit_excess_mwh,
            "optimised_score_gbp": r.per_unit_score_gbp,
            "optimised_gross_gbp": r.gross_gbp,
            "optimised_excess_mwh": r.per_unit_excess_mwh,
            "optimised_throughput_mwh": r.throughput_mwh,
        })
    for rp, rs in zip(pu.itertuples(index=False), sh.itertuples(index=False)):
        rows.append({
            "date": rs.date,
            "partition": "Pillswood+Whitelee+Roosecote",
            "groups": 1,
            "fixed_per_asset_schedule_score_gbp": rp.shared_score_gbp,
            "fixed_per_asset_schedule_gross_gbp": rp.gross_gbp,
            "fixed_per_asset_schedule_excess_mwh": rp.shared_excess_mwh,
            "optimised_score_gbp": rs.shared_score_gbp,
            "optimised_gross_gbp": rs.gross_gbp,
            "optimised_excess_mwh": rs.shared_excess_mwh,
            "optimised_throughput_mwh": rs.throughput_mwh,
        })
    return rows


def rebuild(root: Path, out: Path, workers: int):
    global _G
    dates, price, volume, sched, forecast, fixed_c, fixed_g = _load_parent_arrays(out)
    _G = {"dates": dates.to_numpy(), "price": price, "volume": volume,
          "sched_volume": sched, "forecast": forecast, "fixed_c": fixed_c, "fixed_g": fixed_g}
    workers = max(1, min(workers, len(dates)))
    chunks = [x for x in np.array_split(np.arange(len(dates)), workers) if len(x)]
    if workers == 1:
        middle = _worker(chunks[0])
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            parts = pool.map(_worker, chunks)
        middle = [row for part in parts for row in part]
    rows = _endpoint_rows(out, dates) + middle
    daily = pd.DataFrame(rows).sort_values(["date", "groups", "partition"]).reset_index(drop=True)
    daily.to_csv(out / "partition_assignment_daily.csv.gz", index=False,
                 compression={"method":"gzip","compresslevel":6,"mtime":0})
    summary = daily.groupby(["partition", "groups"], as_index=False).agg(
        fixed_per_asset_schedule_score_gbp=("fixed_per_asset_schedule_score_gbp", "sum"),
        fixed_per_asset_schedule_gross_gbp=("fixed_per_asset_schedule_gross_gbp", "sum"),
        fixed_per_asset_schedule_excess_mwh=("fixed_per_asset_schedule_excess_mwh", "sum"),
        optimised_score_gbp=("optimised_score_gbp", "sum"),
        optimised_gross_gbp=("optimised_gross_gbp", "sum"),
        optimised_excess_mwh=("optimised_excess_mwh", "sum"),
        optimised_throughput_mwh_day=("optimised_throughput_mwh", "mean"),
    )
    order = {name: i for i, name in enumerate(PARTITIONS)}
    summary["_order"] = summary.partition.map(order)
    summary = summary.sort_values("_order").drop(columns="_order")
    summary.to_csv(out / "partition_assignment_summary.csv", index=False)

    fine = summary.iloc[0]; coarse = summary.iloc[-1]
    checks = {
        "daily_rows": int(len(daily)),
        "expected_daily_rows": int(1081 * len(PARTITIONS)),
        "all_fixed_scores_between_endpoints": bool(((summary.fixed_per_asset_schedule_score_gbp <= fine.fixed_per_asset_schedule_score_gbp + 1e-6) & (summary.fixed_per_asset_schedule_score_gbp >= coarse.fixed_per_asset_schedule_score_gbp - 1e-6)).all()),
        "all_optimised_scores_between_endpoints": bool(((summary.optimised_score_gbp <= fine.optimised_score_gbp + 1e-6) & (summary.optimised_score_gbp >= coarse.optimised_score_gbp - 1e-6)).all()),
        "reoptimisation_weakly_improves_matching_audit": bool((summary.optimised_score_gbp >= summary.fixed_per_asset_schedule_score_gbp - 1e-6).all()),
    }
    status = "PASS" if checks["daily_rows"] == checks["expected_daily_rows"] and all(v for k,v in checks.items() if isinstance(v,bool)) else "FAIL"
    report = {"status": status, "checks": checks}
    (out / "partition_assignment_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2))
    if status != "PASS": raise RuntimeError(report)


def validate(out: Path):
    """Validate the stored all-partition results from independent recomputations.

    The theorem applies to a fixed schedule and common audit inputs.  Ordering of
    realised scores from separately reoptimised schedules is recorded as an
    empirical property of this case rather than treated as a theorem.
    """
    tol = 1e-6
    summary = pd.read_csv(out / "partition_assignment_summary.csv")
    daily = pd.read_csv(out / "partition_assignment_daily.csv.gz", parse_dates=["date"])
    stored_report = json.loads((out / "partition_assignment_validation.json").read_text())
    expected = {name: len(blocks) for name, blocks in PARTITIONS.items()}
    failures = []

    if len(summary) != len(expected):
        failures.append(f"summary rows {len(summary)} != {len(expected)}")
    if len(daily) != 1081 * len(expected):
        failures.append(f"daily rows {len(daily)} != {1081 * len(expected)}")
    if set(summary.partition) != set(expected) or set(daily.partition) != set(expected):
        failures.append("partition labels do not match the declared five partitions")
    if daily.duplicated(["date", "partition"]).any():
        failures.append("duplicate date-partition rows")

    counts = daily.groupby("partition").date.nunique().to_dict()
    if any(counts.get(name) != 1081 for name in expected):
        failures.append(f"unexpected date counts by partition: {counts}")
    declared_groups = summary.set_index("partition").groups.to_dict()
    if any(int(declared_groups.get(name, -1)) != groups for name, groups in expected.items()):
        failures.append(f"unexpected group counts: {declared_groups}")

    agg_spec = {
        "fixed_per_asset_schedule_score_gbp": "sum",
        "fixed_per_asset_schedule_gross_gbp": "sum",
        "fixed_per_asset_schedule_excess_mwh": "sum",
        "optimised_score_gbp": "sum",
        "optimised_gross_gbp": "sum",
        "optimised_excess_mwh": "sum",
        "optimised_throughput_mwh": "mean",
    }
    recomputed = daily.groupby(["partition", "groups"], as_index=False).agg(agg_spec)
    recomputed = recomputed.rename(columns={"optimised_throughput_mwh": "optimised_throughput_mwh_day"})
    merged = summary.merge(recomputed, on=["partition", "groups"], suffixes=("_stored", "_recomputed"), validate="one_to_one")
    summary_errors = {}
    for col in [c for c in summary.columns if c not in {"partition", "groups"}]:
        err = float(np.max(np.abs(merged[f"{col}_stored"] - merged[f"{col}_recomputed"])))
        summary_errors[col] = err
        if err > tol:
            failures.append(f"summary mismatch for {col}: {err}")

    fine_name = "Pillswood | Whitelee | Roosecote"
    coarse_name = "Pillswood+Whitelee+Roosecote"
    fixed_pivot = daily.pivot(index="date", columns="partition", values="fixed_per_asset_schedule_score_gbp")
    fine_fixed = fixed_pivot[fine_name]
    coarse_fixed = fixed_pivot[coarse_name]
    fixed_order_ok = bool(((fixed_pivot.le(fine_fixed + tol, axis=0)) & (fixed_pivot.ge(coarse_fixed - tol, axis=0))).all().all())
    if not fixed_order_ok:
        failures.append("fixed-schedule partition-refinement ordering failed")

    gross_span = daily.groupby("date").fixed_per_asset_schedule_gross_gbp.agg(lambda x: float(x.max() - x.min()))
    fixed_gross_max_span = float(gross_span.max())
    if fixed_gross_max_span > tol:
        failures.append(f"fixed-schedule gross marks differ across partitions: {fixed_gross_max_span}")

    summary_by_name = summary.set_index("partition")
    fine_opt = float(summary_by_name.loc[fine_name, "optimised_score_gbp"])
    coarse_opt = float(summary_by_name.loc[coarse_name, "optimised_score_gbp"])
    empirical_opt_between = bool(((summary.optimised_score_gbp <= fine_opt + tol) & (summary.optimised_score_gbp >= coarse_opt - tol)).all())
    empirical_reopt_improves = bool((summary.optimised_score_gbp >= summary.fixed_per_asset_schedule_score_gbp - tol).all())
    if not empirical_opt_between:
        failures.append("aggregate realised optimised scores are not between endpoint scores")
    if not empirical_reopt_improves:
        failures.append("aggregate realised reoptimised score is below its matching fixed-schedule audit")

    # Independently re-solve three sentinel days for each intermediate partition.
    dates, price, volume, sched, forecast, fixed_c, fixed_g = _load_parent_arrays(out)
    sentinel_indices = [0, len(dates)//2, len(dates)-1]
    sentinel_errors = []
    for idx in sentinel_indices:
        date = pd.Timestamp(dates[idx])
        for name, partition in INTERMEDIATE.items():
            row = daily[(daily.date == date) & (daily.partition == name)]
            if len(row) != 1:
                failures.append(f"missing sentinel row {date.date()} / {name}")
                continue
            row = row.iloc[0]
            fixed = audit_partition(fixed_c[idx], fixed_g[idx], price[idx], volume[idx], partition)
            c, g = PartitionLP(partition).solve(forecast[idx], sched[idx])
            opt = audit_partition(c, g, price[idx], volume[idx], partition)
            pairs = {
                "fixed_score_gbp": (row.fixed_per_asset_schedule_score_gbp, fixed["score_gbp"]),
                "fixed_gross_gbp": (row.fixed_per_asset_schedule_gross_gbp, fixed["gross_gbp"]),
                "fixed_excess_mwh": (row.fixed_per_asset_schedule_excess_mwh, fixed["excess_mwh"]),
                "optimised_score_gbp": (row.optimised_score_gbp, opt["score_gbp"]),
                "optimised_gross_gbp": (row.optimised_gross_gbp, opt["gross_gbp"]),
                "optimised_excess_mwh": (row.optimised_excess_mwh, opt["excess_mwh"]),
                "optimised_throughput_mwh": (row.optimised_throughput_mwh, opt["throughput_mwh"]),
            }
            for field, (stored, rebuilt) in pairs.items():
                sentinel_errors.append({
                    "date": str(date.date()), "partition": name, "field": field,
                    "abs_error": float(abs(float(stored) - float(rebuilt)))
                })
    sentinel_max_error = max((x["abs_error"] for x in sentinel_errors), default=0.0)
    if sentinel_max_error > tol:
        failures.append(f"sentinel re-solve mismatch: {sentinel_max_error}")

    checks = {
        "daily_rows": int(len(daily)),
        "expected_daily_rows": int(1081 * len(expected)),
        "unique_dates": int(daily.date.nunique()),
        "fixed_schedule_ordering_all_dates": fixed_order_ok,
        "fixed_schedule_gross_max_span_gbp": fixed_gross_max_span,
        "summary_reaggregation_max_abs_error": max(summary_errors.values(), default=0.0),
        "reported_optimised_summary_scores_between_endpoints": empirical_opt_between,
        "reported_aggregate_reoptimisation_improves_matching_audit": empirical_reopt_improves,
        "sentinel_days": [str(pd.Timestamp(dates[i]).date()) for i in sentinel_indices],
        "sentinel_resolve_max_abs_error": sentinel_max_error,
        "stored_report_status": stored_report.get("status"),
    }
    status = "PASS" if not failures and stored_report.get("status") == "PASS" else "FAIL"
    result = {"status": status, "tolerance": tol, "summary_rows": int(len(summary)), "daily_rows": int(len(daily)), "checks": checks, "failures": failures}
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise RuntimeError(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    args = ap.parse_args()
    root = detect_root(); out = output_root(root)
    if args.rebuild: rebuild(root, out, args.workers)
    if args.validate: validate(out)
    if not args.rebuild and not args.validate: ap.print_help()

if __name__ == "__main__":
    main()
