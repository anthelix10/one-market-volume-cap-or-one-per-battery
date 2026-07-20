#!/usr/bin/env python3
"""Rebuild and validate the public system-price/MID wedge calibration.

The calibration panel contains 52,026 period-level inner-join observations from
1 January 2023 through 22 December 2025. It covers the 1,084 non-spring dates
in that window and omits six period joins for which either the system-price or
MID record is absent (2023-01-17 SP42; 2023-01-22 SP9-11; 2023-03-18 SP9-10).
It is intentionally broader than the 51,888-period controller panel because it
does not require complete PN/B1610 coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summary_from_periods(df: pd.DataFrame) -> dict:
    wedge = df["abs_cashout_mid_spread_GBP_per_MWh"].to_numpy(float)
    low = df.loc[df["MID_depth_decile"] == df["MID_depth_decile"].min(), "abs_cashout_mid_spread_GBP_per_MWh"].to_numpy(float)
    return {
        "matched_periods": int(len(df)),
        "panel_definition": "Inner join of system-price and MID observations over the 1,084 non-spring dates from 2023-01-01 through 2025-12-22; six missing period joins are omitted. This calibration panel is separate from the 1,081-day controller panel.",
        "system_price_mid_abs_wedge_median_GBP_per_MWh": float(np.quantile(wedge, 0.50)),
        "system_price_mid_abs_wedge_p75_GBP_per_MWh": float(np.quantile(wedge, 0.75)),
        "system_price_mid_abs_wedge_p90_GBP_per_MWh": float(np.quantile(wedge, 0.90)),
        "low_depth_decile_abs_wedge_p75_GBP_per_MWh": float(np.quantile(low, 0.75)),
        "chosen_base_phi_GBP_per_MWh": 30.0,
        "calibration_interpretation": "Base phi = GBP 30/MWh is a rounded public-stack reference close to the 75th percentile of |system price - MID price|, not a private execution-cost estimate.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-stored", action="store_true")
    ap.add_argument("--write", action="store_true", help="Rewrite the stored one-row CSV")
    ap.add_argument("--report", type=Path, help="Optional JSON validation-report path")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    periods_path = root / "inputs/extensions/system_price_mid_phi_calibration_periods.csv"
    table_path = root / "inputs/extensions/phi_reference_calibration_summary.csv"

    df = pd.read_csv(periods_path)
    calc = summary_from_periods(df)
    if args.write:
        pd.DataFrame([calc]).to_csv(table_path, index=False)
    if args.validate_stored or not args.write:
        stored = pd.read_csv(table_path).iloc[0].to_dict()
        errors = {}
        for key, value in calc.items():
            if isinstance(value, (int, float)):
                errors[key] = abs(float(value) - float(stored[key]))
            else:
                errors[key] = 0 if str(value) == str(stored[key]) else 1
        max_error = max(float(v) for v in errors.values())
        report = {"status": "PASS" if max_error <= 1e-9 else "FAIL", "max_error": max_error, "errors": errors, "summary": calc}
        report_text=json.dumps(report, indent=2)
        if args.report:
            args.report.write_text(report_text + "\n", encoding="utf-8")
        print(report_text)
        if report["status"] != "PASS":
            raise SystemExit(1)
    else:
        print(json.dumps(calc, indent=2))


if __name__ == "__main__":
    main()
