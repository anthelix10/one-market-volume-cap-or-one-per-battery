#!/usr/bin/env python3
"""Verify the analytical inputs against the archived Elexon responses.

The command is read-only. It parses all 702 JSON response bodies, checks the
request list and returned record counts, reconstructs the MID pivots and the PN
and B1610 asset-period panels, and compares them with ``inputs/current_snapshot``.

Example
-------
python code/verify_inputs_from_raw_archive.py \
  --raw-archive ../elexon_mid_pn_b1610_raw_responses_2023_2025.zip
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ASSET_MAP = {
    "E_PILLB-1": "Pillswood Battery Storage",
    "E_PILLB-2": "Pillswood Battery Storage",
    "T_WHLWB-1": "Whitelee 1 Battery",
    "E_ROOSB-1": "Roosecote Battery",
}
PERIODS = list(range(1, 49))


def read_csv_member(archive: zipfile.ZipFile, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(archive.read(name)), **kwargs)


def load_raw_tables(
    archive: zipfile.ZipFile,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, dict[str, int], list[str]]:
    """Parse the archived responses and check their request-list entries."""
    requests = read_csv_member(archive, "provenance/REQUESTS.csv")
    members = set(archive.namelist())
    rows: dict[str, list[dict]] = defaultdict(list)
    errors: list[str] = []
    counts: dict[str, int] = defaultdict(int)

    for request in requests.itertuples(index=False):
        member = str(request.file).replace("\\", "/")
        if member not in members:
            errors.append(f"missing raw response: {member}")
            continue
        payload = archive.read(member)
        try:
            response = json.loads(payload)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON response: {member}: {exc}")
            continue
        if not isinstance(response, list):
            errors.append(f"response is not a JSON list: {member}")
            continue
        if len(response) != int(request.record_count):
            errors.append(f"record-count mismatch: {member}")
        dataset = str(request.dataset)
        counts[dataset] += len(response)
        rows[dataset].extend(response)

    expected_datasets = {"MID", "PN", "B1610"}
    missing_datasets = expected_datasets.difference(rows)
    if missing_datasets:
        errors.append(f"missing datasets: {sorted(missing_datasets)}")

    mid = pd.DataFrame(rows["MID"]).rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
        }
    )
    pn = pd.DataFrame(rows["PN"]).rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "timeFrom": "time_from",
            "timeTo": "time_to",
            "levelFrom": "level_from",
            "levelTo": "level_to",
            "bmUnit": "bm_unit",
        }
    )
    b1610 = pd.DataFrame(rows["B1610"]).rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "bmUnit": "bm_unit",
        }
    )
    return mid, pn, b1610, len(requests), dict(counts), errors


def rebuild_mid(mid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mid = mid[mid["settlement_period"].between(1, 48)].copy()
    mid["settlement_date"] = pd.to_datetime(mid["settlement_date"])
    mid["vp"] = mid["price"] * mid["volume"]
    grouped = mid.groupby(["settlement_date", "settlement_period"], as_index=False).agg(
        volume=("volume", "sum"),
        vp=("vp", "sum"),
        mean_price=("price", "mean"),
    )
    grouped["price"] = np.where(
        grouped["volume"] > 0,
        grouped["vp"] / grouped["volume"],
        grouped["mean_price"],
    )
    price = grouped.pivot(index="settlement_date", columns="settlement_period", values="price")
    volume = grouped.pivot(index="settlement_date", columns="settlement_period", values="volume")
    for settlement_period in PERIODS:
        if settlement_period not in price.columns:
            price[settlement_period] = np.nan
            volume[settlement_period] = np.nan
    return price[PERIODS].sort_index(), volume[PERIODS].sort_index()


def rebuild_pn(pn: pd.DataFrame) -> pd.DataFrame:
    pn = pn.copy()
    pn["settlement_date"] = pd.to_datetime(pn["settlement_date"])
    pn["time_from"] = pd.to_datetime(pn["time_from"], utc=True)
    pn["time_to"] = pd.to_datetime(pn["time_to"], utc=True)
    pn["duration_hours"] = (pn["time_to"] - pn["time_from"]).dt.total_seconds() / 3600.0
    pn["segment_mwh"] = 0.5 * (pn["level_from"] + pn["level_to"]) * pn["duration_hours"]
    unit = pn.groupby(["bm_unit", "settlement_date", "settlement_period"], as_index=False).agg(
        pnMWh=("segment_mwh", "sum"),
        coverageMinutes=("duration_hours", lambda values: float(values.sum() * 60.0)),
        segmentCount=("segment_mwh", "size"),
    )
    unit["asset"] = unit["bm_unit"].map(ASSET_MAP)
    if unit["asset"].isna().any():
        raise RuntimeError("unexpected BM unit in PN responses")
    return unit.groupby(["asset", "settlement_date", "settlement_period"], as_index=False).agg(
        pnMWh=("pnMWh", "sum"),
        coverageMinutes=("coverageMinutes", "sum"),
        segmentCount=("segmentCount", "sum"),
    )


def rebuild_b1610(b1610: pd.DataFrame) -> pd.DataFrame:
    b1610 = b1610.copy()
    b1610["settlement_date"] = pd.to_datetime(b1610["settlement_date"])
    b1610["asset"] = b1610["bm_unit"].map(ASSET_MAP)
    if b1610["asset"].isna().any():
        raise RuntimeError("unexpected BM unit in B1610 responses")
    return b1610.groupby(["asset", "settlement_date", "settlement_period"], as_index=False).agg(
        quantity=("quantity", "sum")
    )


def max_pivot_error(calculated: pd.DataFrame, stored_path: Path) -> tuple[float, int]:
    stored = pd.read_csv(stored_path, parse_dates=["date"]).set_index("date")
    stored.columns = [int(column) for column in stored.columns]
    dates = stored.index.union(calculated.index)
    left = calculated.reindex(dates)[PERIODS]
    right = stored.reindex(dates)[PERIODS]
    missing = int((left.isna() ^ right.isna()).to_numpy().sum())
    common = left.notna() & right.notna()
    error = float(
        np.nanmax(np.abs(left.where(common).to_numpy() - right.where(common).to_numpy()))
    )
    return error, missing


def max_panel_error(
    calculated: pd.DataFrame, stored_path: Path, value: str
) -> tuple[float, int, int]:
    stored = pd.read_csv(stored_path, parse_dates=["settlementDate"])
    stored = stored.rename(
        columns={
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
        }
    )
    merged = stored[["asset", "settlement_date", "settlement_period", value]].merge(
        calculated[["asset", "settlement_date", "settlement_period", value]],
        on=["asset", "settlement_date", "settlement_period"],
        how="outer",
        suffixes=("_stored", "_calculated"),
        indicator=True,
    )
    common = merged[merged["_merge"] == "both"]
    error = (
        float(
            np.max(
                np.abs(common[f"{value}_stored"] - common[f"{value}_calculated"])
            )
        )
        if len(common)
        else float("inf")
    )
    return (
        error,
        int((merged["_merge"] == "left_only").sum()),
        int((merged["_merge"] == "right_only").sum()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-archive", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.raw_archive) as archive:
        mid, pn, b1610, request_count, record_counts, request_errors = load_raw_tables(archive)

    price, volume = rebuild_mid(mid)
    pn_panel = rebuild_pn(pn)
    b1610_panel = rebuild_b1610(b1610)

    price_error, price_missing = max_pivot_error(
        price, root / "inputs/current_snapshot/current_mid_price_pivot_2023_2025.csv"
    )
    volume_error, volume_missing = max_pivot_error(
        volume, root / "inputs/current_snapshot/current_mid_volume_pivot_2023_2025.csv"
    )
    pn_error, pn_stored_only, pn_raw_only = max_panel_error(
        pn_panel,
        root / "inputs/current_snapshot/current_latest_pn_asset_period_panel.csv",
        "pnMWh",
    )
    b1610_error, b1610_stored_only, b1610_raw_only = max_panel_error(
        b1610_panel,
        root / "inputs/current_snapshot/current_latest_b1610_asset_period_panel.csv",
        "quantity",
    )

    report = {
        "status": "PASS",
        "raw_archive": args.raw_archive.name,
        "request_rows": request_count,
        "record_counts": record_counts,
        "request_level_errors": request_errors,
        "comparisons": {
            "mid_price_max_abs_error": price_error,
            "mid_price_missing_cell_mismatches": price_missing,
            "mid_volume_max_abs_error": volume_error,
            "mid_volume_missing_cell_mismatches": volume_missing,
            "pn_mwh_max_abs_error": pn_error,
            "pn_stored_only_rows": pn_stored_only,
            "pn_raw_only_rows": pn_raw_only,
            "b1610_mwh_max_abs_error": b1610_error,
            "b1610_stored_only_rows": b1610_stored_only,
            "b1610_raw_only_rows": b1610_raw_only,
        },
        "interpretation": (
            "Raw-only rows are excluded clock-change or incomplete-panel dates; "
            "every stored analytical row must match the response-derived reconstruction."
        ),
    }
    failures = list(request_errors)
    if price_missing or volume_missing or pn_stored_only or b1610_stored_only:
        failures.append("stored analytical inputs are missing from the response-derived reconstruction")
    if max(price_error, volume_error, pn_error, b1610_error) > args.tolerance:
        failures.append("numerical mismatch exceeds tolerance")
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
