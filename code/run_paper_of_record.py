#!/usr/bin/env python3
"""Rebuild and validate the 1,081-day BESS analysis.

The script implements the independent per-asset linear programmes and the joint
aggregate-cap formulation used in the article. It uses the deposited processed
MID, PN and B1610 inputs, with the 2022 MID files used only for forecast and cap
warm-up.

Commands
--------
  --rebuild-reference      Rebuild the reference schedules and summaries.
  --reoptimise-grid        Rebuild the 25-cell aggregate-cap surface.
  --rebuild-forecasts      Rebuild the declared forecast sensitivity.
  --rebuild-operational    Rebuild the PN, B1610 and BOALF descriptive summary.
  --rebuild-all            Run the four rebuild steps above.
  --validate-stored        Validate the retained stored outputs and sentinel LPs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DELTA = 0.5
ETA_C = 0.93
ETA_D = 0.93
BASE_RHO = 0.02
BASE_PHI = 30.0
BASE_TRAINING_DAYS = 364
BASE_ALPHA = 20.0
PERIODS = 48
RHO_GRID = [0.005, 0.01, 0.02, 0.03, 0.05]
PHI_GRID = [10.0, 20.0, 30.0, 45.0, 60.0]
RIDGE_LOOKBACKS = [91, 182, 364, 730]
RIDGE_ALPHAS = [0.1, 1.0, 5.0, 20.0, 100.0]
DAILY_TIE_TOLERANCE_GBP = 1.0
GZIP_COMPRESSION = {"method": "gzip", "compresslevel": 6, "mtime": 0}

CONTINUOUS = [
    "lag1", "lag2", "lag7", "roll7_mean_sp", "roll7_std_sp",
    "roll28_mean_sp", "roll28_std_sp", "prev_day_mean", "prev_day_max",
    "prev_day_min", "prev_day_spread", "roll7_daily_mean",
    "roll28_daily_mean", "weekend",
]
CATEGORICAL = ["period_cat", "dow_cat", "month_cat"]
FEATURES = CONTINUOUS + CATEGORICAL

@dataclass(frozen=True)
class Asset:
    name: str
    bm_unit: str
    pmax: float
    emax: float

ASSETS = [
    Asset("Pillswood Battery Storage", "E_PILLB-1 + E_PILLB-2", 98.0, 196.0),
    Asset("Whitelee 1 Battery", "T_WHLWB-1", 50.0, 50.0),
    Asset("Roosecote Battery", "E_ROOSB-1", 49.0, 24.5),
]
ASSET_INDEX = {a.name: i for i, a in enumerate(ASSETS)}
PERIOD_COLS = [str(i) for i in range(1, 49)]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def detect_root() -> Path:
    here = Path(__file__).resolve()
    # Installed in supplementary/code.
    if (here.parents[1] / "inputs" / "current_snapshot").exists():
        return here.parents[1]
    # Development tree: code next to source supplement.
    env = os.environ.get("BESS_SUPPLEMENT_ROOT")
    if env:
        return Path(env)
    candidate = Path.cwd()
    if (candidate / "inputs" / "current_snapshot").exists():
        return candidate
    raise RuntimeError("Could not locate supplementary package root")


def output_root(root: Path) -> Path:
    env = os.environ.get("BESS_OUTPUT_ROOT")
    if env:
        p = Path(env)
    else:
        p = root / "paper_of_record"
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_pivot(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def build_current_inputs(root: Path, out: Path) -> dict:
    """Build the current-snapshot complete MID panel and matched dates."""
    warmup_price_source = read_pivot(root / "inputs" / "warmup" / "mid_price_warmup_2022.csv")
    warmup_volume_source = read_pivot(root / "inputs" / "warmup" / "mid_volume_warmup_2022.csv")
    current_price = read_pivot(root / "inputs" / "current_snapshot" / "current_mid_price_pivot_2023_2025.csv")
    current_volume = read_pivot(root / "inputs" / "current_snapshot" / "current_mid_volume_pivot_2023_2025.csv")

    warm_price = warmup_price_source.loc[warmup_price_source.index < pd.Timestamp("2023-01-01"), PERIOD_COLS]
    warm_volume = warmup_volume_source.loc[warmup_volume_source.index < pd.Timestamp("2023-01-01"), PERIOD_COLS]
    combined_price = pd.concat([warm_price, current_price[PERIOD_COLS]]).sort_index()
    combined_volume = pd.concat([warm_volume, current_volume[PERIOD_COLS]]).sort_index()

    # Forecast features use complete 48-period days only.
    complete_price = combined_price.dropna(subset=PERIOD_COLS)
    complete_volume = combined_volume.dropna(subset=PERIOD_COLS)
    common_complete = complete_price.index.intersection(complete_volume.index)
    complete_price = complete_price.loc[common_complete]
    complete_volume = complete_volume.loc[common_complete]

    pn = pd.read_csv(
        root / "inputs" / "current_snapshot" / "current_latest_pn_asset_period_panel.csv",
        parse_dates=["settlementDate"],
    )
    counts = pn.groupby(["settlementDate", "asset"])["settlementPeriod"].nunique().unstack(fill_value=0)
    complete_pn = counts[(counts >= PERIODS).all(axis=1)].index
    eval_start, eval_end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-22")
    matched = pd.DatetimeIndex(
        sorted(
            d for d in complete_pn
            if eval_start <= d <= eval_end and d in common_complete
        )
    )
    if len(matched) != 1081:
        raise RuntimeError(f"Expected 1,081 matched days, found {len(matched)}")

    # The one-day-shifted 28-day same-period median; every evaluation day has a full window.
    sched_volume = complete_volume[PERIOD_COLS].shift(1).rolling(28, min_periods=28).median()
    if sched_volume.loc[matched].isna().any().any():
        raise RuntimeError("Scheduling volume contains missing values in the evaluation window")

    # Persist canonical analytical inputs.
    complete_price.reset_index().to_csv(out / "mid_price_complete_2022_2025.csv", index=False)
    complete_volume.reset_index().to_csv(out / "mid_volume_complete_2022_2025.csv", index=False)
    sched_volume.reset_index().to_csv(out / "mid_volume_trailing28_median.csv", index=False)
    pd.DataFrame({"date": matched}).to_csv(out / "matched_dates_1081.csv", index=False)

    exclusion_rows = [
        {"date": "2023-03-26", "reason": "spring clock-change day (46 periods)"},
        {"date": "2024-03-31", "reason": "spring clock-change day (46 periods)"},
        {"date": "2025-03-30", "reason": "spring clock-change day (46 periods)"},
        {"date": "2023-07-17", "reason": "incomplete PN coverage"},
        {"date": "2023-12-29", "reason": "incomplete PN coverage"},
        {"date": "2023-06-07", "reason": "current MID missing settlement period 38"},
    ]
    panel = {
        "candidate_days": 1087,
        "matched_days": len(matched),
        "periods": len(matched) * PERIODS,
        "study_start": str(matched.min().date()),
        "study_end": str(matched.max().date()),
        "exclusions": exclusion_rows,
        "warmup_start": str(complete_price.index.min().date()),
        "scheduling_window_days": 28,
        "evaluation_days_with_incomplete_scheduling_window": 0,
        "current_snapshot_sources": ["MID", "PN", "B1610"],
    }
    write_json(out / "panel_construction_1081.json", panel)

    return {
        "price": complete_price,
        "volume": complete_volume,
        "sched_volume": sched_volume,
        "matched": matched,
        "pn": pn,
        "current_price": current_price,
        "current_volume": current_volume,
        "panel": panel,
    }


def build_features(price: pd.DataFrame) -> pd.DataFrame:
    x = price[PERIOD_COLS].reset_index()
    long = x.melt(id_vars="date", var_name="period", value_name="y")
    long["period"] = long["period"].astype(int)
    long = long.sort_values(["period", "date"])
    g = long.groupby("period")["y"]
    for lag in [1, 2, 7]:
        long[f"lag{lag}"] = g.shift(lag)
    long["roll7_mean_sp"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    long["roll7_std_sp"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=7).std(ddof=0))
    long["roll28_mean_sp"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    long["roll28_std_sp"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=28).std(ddof=0))
    daily = price[PERIOD_COLS]
    dm, dx, dn = daily.mean(axis=1), daily.max(axis=1), daily.min(axis=1)
    ds = dx - dn
    ddf = pd.DataFrame({
        "date": daily.index,
        "prev_day_mean": dm.shift(1).values,
        "prev_day_max": dx.shift(1).values,
        "prev_day_min": dn.shift(1).values,
        "prev_day_spread": ds.shift(1).values,
        "roll7_daily_mean": dm.shift(1).rolling(7, min_periods=7).mean().values,
        "roll28_daily_mean": dm.shift(1).rolling(28, min_periods=28).mean().values,
    })
    long = long.merge(ddf, on="date")
    long["weekend"] = (long["date"].dt.dayofweek >= 5).astype(float)
    long["period_cat"] = long["period"].astype(int)
    long["dow_cat"] = long["date"].dt.dayofweek.astype(int)
    long["month_cat"] = long["date"].dt.month.astype(int)
    cols = ["date", "period", "y", *CONTINUOUS, *CATEGORICAL]
    return long[cols].dropna().sort_values(["date", "period"]).reset_index(drop=True)


def rolling_ridge(feature_panel: pd.DataFrame, dates: pd.DatetimeIndex, training_days: int, alpha: float) -> pd.DataFrame:
    eval_set = set(dates)
    preds = []
    for month_start in pd.date_range(dates.min().replace(day=1), dates.max().replace(day=1), freq="MS"):
        month_end = month_start + pd.offsets.MonthEnd(0)
        test_mask = (
            (feature_panel.date >= month_start)
            & (feature_panel.date <= month_end)
            & feature_panel.date.isin(eval_set)
        )
        if not test_mask.any():
            continue
        train_start = month_start - pd.Timedelta(days=training_days)
        train = feature_panel[(feature_panel.date >= train_start) & (feature_panel.date < month_start)]
        test = feature_panel[test_mask]
        pre = ColumnTransformer([
            ("continuous", StandardScaler(), CONTINUOUS),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])
        model = make_pipeline(pre, Ridge(alpha=alpha))
        model.fit(train[FEATURES], train.y)
        o = test[["date", "period", "y"]].copy()
        o["forecast_price"] = model.predict(test[FEATURES])
        preds.append(o)
    return pd.concat(preds, ignore_index=True).sort_values(["date", "period"]).reset_index(drop=True)


def rolling_gbm(feature_panel: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    eval_set = set(dates)
    preds = []
    for month_start in pd.date_range(dates.min().replace(day=1), dates.max().replace(day=1), freq="MS"):
        month_end = month_start + pd.offsets.MonthEnd(0)
        test_mask = (
            (feature_panel.date >= month_start)
            & (feature_panel.date <= month_end)
            & feature_panel.date.isin(eval_set)
        )
        if not test_mask.any():
            continue
        train_start = month_start - pd.Timedelta(days=BASE_TRAINING_DAYS)
        train = feature_panel[(feature_panel.date >= train_start) & (feature_panel.date < month_start)]
        test = feature_panel[test_mask]
        model = GradientBoostingRegressor(
            n_estimators=12, learning_rate=0.08, max_depth=2,
            subsample=0.8, random_state=42, loss="squared_error",
        )
        model.fit(train[FEATURES], train.y)
        o = test[["date", "period", "y"]].copy()
        o["forecast_price"] = model.predict(test[FEATURES])
        preds.append(o)
    return pd.concat(preds, ignore_index=True).sort_values(["date", "period"]).reset_index(drop=True)


def forecast_metrics(fc: pd.DataFrame) -> dict:
    err = fc.forecast_price.to_numpy() - fc.y.to_numpy()
    d = fc.groupby("date").agg(
        realized_mean=("y", "mean"), forecast_mean=("forecast_price", "mean"),
        realized_spread=("y", lambda x: float(x.max() - x.min())),
        forecast_spread=("forecast_price", lambda x: float(x.max() - x.min())),
    )
    return {
        "period_mae_gbp_mwh": float(np.mean(np.abs(err))),
        "period_rmse_gbp_mwh": float(np.sqrt(np.mean(err**2))),
        "period_bias_gbp_mwh": float(np.mean(err)),
        "daily_mean_price_correlation": float(d.realized_mean.corr(d.forecast_mean)),
        "daily_spread_correlation": float(d.realized_spread.corr(d.forecast_spread)),
        "daily_spread_mae_gbp_mwh": float(np.mean(np.abs(d.forecast_spread - d.realized_spread))),
        "evaluation_days": int(d.shape[0]),
        "evaluation_periods": int(len(fc)),
    }


class SingleLP:
    def __init__(self, asset: Asset, kind: str):
        self.asset, self.kind = asset, kind
        self.over = kind == "priced"
        n = PERIODS
        self.nvars = 2*n + (2*n if self.over else 0)
        rows, cols, data, b = [], [], [], []
        rowno = 0
        # SOC cumulative bounds.
        for t in range(1, n+1):
            for sign, rhs in [(1.0, asset.emax/2.0), (-1.0, asset.emax/2.0)]:
                # sign=1 cumulative <= E/2; sign=-1 -cumulative <= E/2.
                for k in range(t):
                    rows.extend([rowno, rowno])
                    cols.extend([k, n+k])
                    data.extend([sign*ETA_C*DELTA, sign*(-DELTA/ETA_D)])
                b.append(rhs)
                rowno += 1
        # Priced slacks against scheduling cap.
        self.cap_rows = []
        if self.over:
            ec, eg = 2*n, 3*n
            for t in range(n):
                rows.extend([rowno,rowno]); cols.extend([t,ec+t]); data.extend([1.0,-1.0]); b.append(0.0); self.cap_rows.append(rowno); rowno += 1
                rows.extend([rowno,rowno]); cols.extend([n+t,eg+t]); data.extend([1.0,-1.0]); b.append(0.0); self.cap_rows.append(rowno); rowno += 1
        self.Aub = sparse.csr_matrix((data,(rows,cols)),shape=(rowno,self.nvars))
        self.base_b = np.asarray(b,float)
        # Daily neutrality.
        eq_cols, eq_data = [], []
        for t in range(n):
            eq_cols.extend([t,n+t]); eq_data.extend([ETA_C*DELTA,-DELTA/ETA_D])
        self.Aeq = sparse.csr_matrix((eq_data,([0]*len(eq_cols),eq_cols)),shape=(1,self.nvars))
        self.beq = np.array([0.0])

    def solve(self, forecast: np.ndarray, cap: np.ndarray | None, phi: float = BASE_PHI) -> tuple[np.ndarray,np.ndarray]:
        n=PERIODS
        obj=np.zeros(self.nvars)
        obj[:n]=DELTA*forecast
        obj[n:2*n]=-DELTA*forecast
        if self.over:
            obj[2*n:4*n]=phi*DELTA
        if self.kind=="hard":
            bounds=[(0.0,float(cap[t])) for t in range(n)]*2
        else:
            bounds=[(0.0,self.asset.pmax)]*(2*n)
            if self.over: bounds += [(0.0,None)]*(2*n)
        b=self.base_b.copy()
        if self.over:
            b[np.asarray(self.cap_rows)] = np.repeat(cap,2)
        r=linprog(obj,A_ub=self.Aub,b_ub=b,A_eq=self.Aeq,b_eq=self.beq,bounds=bounds,method="highs")
        if not r.success: raise RuntimeError(r.message)
        return r.x[:n],r.x[n:2*n]


class SharedLP:
    def __init__(self, kind: str):
        self.kind=kind; self.over=kind=="priced"
        A=len(ASSETS); n=PERIODS
        self.c0=0; self.g0=A*n; self.ec0=2*A*n; self.eg0=self.ec0+n
        self.nvars=2*A*n+(2*n if self.over else 0)
        rows=[];cols=[];data=[];b=[];rowno=0
        # Per-asset SOC cumulative inequalities.
        for asset_index,a in enumerate(ASSETS):
            for t in range(1,n+1):
                for sign,rhs in [(1.0,a.emax/2.0),(-1.0,a.emax/2.0)]:
                    for k in range(t):
                        ci=asset_index*n+k; gi=self.g0+asset_index*n+k
                        rows.extend([rowno,rowno]);cols.extend([ci,gi]);data.extend([sign*ETA_C*DELTA,sign*(-DELTA/ETA_D)])
                    b.append(rhs);rowno+=1
        # Shared cap rows.
        self.cap_rows=[]
        for t in range(n):
            for side in ["c","g"]:
                for asset_index in range(A):
                    idx=(asset_index*n+t) if side=="c" else (self.g0+asset_index*n+t)
                    rows.append(rowno);cols.append(idx);data.append(1.0)
                if self.over:
                    sidx=(self.ec0+t) if side=="c" else (self.eg0+t)
                    rows.append(rowno);cols.append(sidx);data.append(-1.0)
                b.append(0.0);self.cap_rows.append(rowno);rowno+=1
        self.Aub=sparse.csr_matrix((data,(rows,cols)),shape=(rowno,self.nvars));self.base_b=np.asarray(b,float)
        # Per-asset daily neutrality.
        er=[];ec=[];ed=[]
        for asset_index in range(A):
            for t in range(n):
                er.extend([asset_index,asset_index]);ec.extend([asset_index*n+t,self.g0+asset_index*n+t]);ed.extend([ETA_C*DELTA,-DELTA/ETA_D])
        self.Aeq=sparse.csr_matrix((ed,(er,ec)),shape=(A,self.nvars));self.beq=np.zeros(A)
        bounds=[]
        for a in ASSETS: bounds += [(0.0,a.pmax)]*n
        for a in ASSETS: bounds += [(0.0,a.pmax)]*n
        if self.over: bounds += [(0.0,None)]*(2*n)
        self.bounds=bounds

    def solve(self, forecast: np.ndarray, cap: np.ndarray, phi: float=BASE_PHI) -> tuple[np.ndarray,np.ndarray]:
        A=len(ASSETS);n=PERIODS
        obj=np.zeros(self.nvars)
        for asset_index in range(A):
            obj[asset_index*n:(asset_index+1)*n]=DELTA*forecast
            obj[self.g0+asset_index*n:self.g0+(asset_index+1)*n]=-DELTA*forecast
        if self.over:
            obj[self.ec0:self.ec0+n]=phi*DELTA;obj[self.eg0:self.eg0+n]=phi*DELTA
        b=self.base_b.copy();b[np.asarray(self.cap_rows)]=np.repeat(cap,2)
        r=linprog(obj,A_ub=self.Aub,b_ub=b,A_eq=self.Aeq,b_eq=self.beq,bounds=self.bounds,method="highs")
        if not r.success: raise RuntimeError(r.message)
        return r.x[:A*n].reshape(A,n),r.x[self.g0:self.g0+A*n].reshape(A,n)


def per_unit_cap(volume: np.ndarray, asset: Asset, rho: float) -> np.ndarray:
    return np.minimum(asset.pmax, rho*volume/DELTA)


def shared_cap(volume: np.ndarray, rho: float) -> np.ndarray:
    return np.minimum(sum(a.pmax for a in ASSETS), rho*volume/DELTA)


def evaluate(c: np.ndarray, g: np.ndarray, price: np.ndarray, volume: np.ndarray, rho: float, phi: float) -> dict:
    """Evaluate both per-unit and shared conventions. c/g shape A x 48."""
    gross=float(np.sum(DELTA*price[None,:]*(g-c)))
    throughput=float(np.sum(DELTA*(g+c)))
    pue_c=pue_g=0.0
    for asset_index,a in enumerate(ASSETS):
        cap=per_unit_cap(volume,a,rho)
        pue_c += float(np.sum(DELTA*np.maximum(c[asset_index]-cap,0.0)))
        pue_g += float(np.sum(DELTA*np.maximum(g[asset_index]-cap,0.0)))
    scap=shared_cap(volume,rho)
    sumc=c.sum(axis=0);sumg=g.sum(axis=0)
    she_c=float(np.sum(DELTA*np.maximum(sumc-scap,0.0)))
    she_g=float(np.sum(DELTA*np.maximum(sumg-scap,0.0)))
    return {
        "gross_gbp":gross,
        "throughput_mwh":throughput,
        "per_unit_charge_excess_mwh":pue_c,
        "per_unit_discharge_excess_mwh":pue_g,
        "per_unit_excess_mwh":pue_c+pue_g,
        "per_unit_score_gbp":gross-phi*(pue_c+pue_g),
        "shared_charge_excess_mwh":she_c,
        "shared_discharge_excess_mwh":she_g,
        "shared_excess_mwh":she_c+she_g,
        "shared_score_gbp":gross-phi*(she_c+she_g),
    }


def pn_arrays(pn: pd.DataFrame, date: pd.Timestamp) -> tuple[np.ndarray,np.ndarray]:
    day=pn[(pn.settlementDate==date)&pn.settlementPeriod.between(1,48)]
    c=np.zeros((len(ASSETS),PERIODS));g=np.zeros_like(c)
    for asset_index,a in enumerate(ASSETS):
        d=day[day.asset==a.name].sort_values("settlementPeriod")
        if len(d)!=PERIODS: raise RuntimeError(f"PN incomplete {date} {a.name} {len(d)}")
        m=d.pnMWh.to_numpy(float)
        c[asset_index]=np.maximum(-m,0.0)/DELTA;g[asset_index]=np.maximum(m,0.0)/DELTA
    return c,g


def forecast_map(fc: pd.DataFrame) -> dict[pd.Timestamp,np.ndarray]:
    return {pd.Timestamp(d):g.sort_values("period").forecast_price.to_numpy(float) for d,g in fc.groupby("date")}



def _solve_reference_chunk(payload):
    """Worker for a bounded chunk of dates; restarting workers avoids HiGHS slowdown."""
    start, date_values, fmat, pmat, vmat, svmat, pnc, png = payload
    labels=["PN declaration","Per-unit unrestricted","Per-unit hard","Per-unit priced","Shared hard","Shared priced"]
    single={a.name:{k:SingleLP(a,k) for k in ["unrestricted","hard","priced"]} for a in ASSETS}
    shared={k:SharedLP(k) for k in ["hard","priced"]}
    nday=len(date_values);cs=np.zeros((nday,len(labels),len(ASSETS),PERIODS),dtype=float);gs=np.zeros_like(cs)
    daily=[];overlap_rows=[];cleaned=[]
    for j,datev in enumerate(date_values):
        date=pd.Timestamp(datev);f=fmat[j];rp=pmat[j];rv=vmat[j];sv=svmat[j]
        schedules={"PN declaration":(pnc[j],png[j])}
        for kind,label in [("unrestricted","Per-unit unrestricted"),("hard","Per-unit hard"),("priced","Per-unit priced")]:
            c=np.zeros((len(ASSETS),PERIODS));g=np.zeros_like(c)
            for asset_index,a in enumerate(ASSETS):
                cap=per_unit_cap(sv,a,BASE_RHO);cc,gg=single[a.name][kind].solve(f,cap if kind!="unrestricted" else None,BASE_PHI);c[asset_index]=cc;g[asset_index]=gg
            schedules[label]=(c,g)
        cap=shared_cap(sv,BASE_RHO);schedules["Shared hard"]=shared["hard"].solve(f,cap,BASE_PHI);schedules["Shared priced"]=shared["priced"].solve(f,cap,BASE_PHI)
        for li,label in enumerate(labels):
            c,g=schedules[label];cs[j,li]=c;gs[j,li]=g;m=evaluate(c,g,rp,rv,BASE_RHO,BASE_PHI);daily.append({"date":date,"schedule":label,**m})
            if label in ["Per-unit priced","Shared priced"]:
                overlap=np.minimum(c,g);ii=np.argwhere(overlap>1e-7)
                for asset_index,t in ii:
                    overlap_rows.append({"date":date,"settlement_period":int(t+1),"asset":ASSETS[int(asset_index)].name,"schedule":label,"charge_mw":float(c[asset_index,t]),"discharge_mw":float(g[asset_index,t]),"overlap_mw":float(overlap[asset_index,t]),"overlap_mwh":float(DELTA*overlap[asset_index,t]),"forecast_price_gbp_mwh":float(f[t]),"realized_mid_price_gbp_mwh":float(rp[t]),"negative_forecast_price":bool(f[t]<0),"negative_realized_price":bool(rp[t]<0),"absolute_price_value_bound_gbp":float(DELTA*overlap[asset_index,t]*abs(rp[t]))})
                cm=evaluate(c-overlap,g-overlap,rp,rv,BASE_RHO,BASE_PHI);cleaned.append({"date":date,"schedule":label,**cm})
    return start,cs,gs,daily,overlap_rows,cleaned

def rebuild_reference(root: Path, out: Path, inputs: dict | None=None) -> dict:
    """Rebuild all six reference schedules on the 1,081-day current panel."""
    import multiprocessing as mp
    t0=time.time(); inputs=inputs or build_current_inputs(root,out)
    price,volume,svol,dates,pn=inputs["price"],inputs["volume"],inputs["sched_volume"],inputs["matched"],inputs["pn"]
    feature_path=out/"feature_panel_current.csv.gz"; forecast_path=out/"forecast_base_ridge_period.csv"
    if os.environ.get("BESS_REUSE_FORECAST")=="1" and feature_path.exists() and forecast_path.exists():
        features=pd.read_csv(feature_path,parse_dates=["date"]);fc=pd.read_csv(forecast_path,parse_dates=["date"])
    else:
        features=build_features(price);features.to_csv(feature_path,index=False,float_format="%.17g",compression=GZIP_COMPRESSION);fc=rolling_ridge(features,dates,BASE_TRAINING_DAYS,BASE_ALPHA);fc.to_csv(forecast_path,index=False)
    fwide=fc.pivot(index="date",columns="period",values="forecast_price").loc[dates,range(1,49)].to_numpy(float)
    pmat=price.loc[dates,PERIOD_COLS].to_numpy(float);vmat=volume.loc[dates,PERIOD_COLS].to_numpy(float);svmat=svol.loc[dates,PERIOD_COLS].to_numpy(float)
    pnc=np.zeros((len(dates),len(ASSETS),PERIODS));png=np.zeros_like(pnc)
    for i,d in enumerate(dates): pnc[i],png[i]=pn_arrays(pn,d)
    labels=["PN declaration","Per-unit unrestricted","Per-unit hard","Per-unit priced","Shared hard","Shared priced"]
    c_store=np.zeros((len(dates),len(labels),len(ASSETS),PERIODS),dtype=float);g_store=np.zeros_like(c_store)
    daily_rows=[];overlap_rows=[];cleaned=[]
    chunk=50;tasks=[]
    for st in range(0,len(dates),chunk):
        en=min(st+chunk,len(dates));tasks.append((st,dates[st:en].to_numpy(),fwide[st:en],pmat[st:en],vmat[st:en],svmat[st:en],pnc[st:en],png[st:en]))
    workers=min(8,os.cpu_count() or 2)
    ctx=mp.get_context("spawn" if os.name == "nt" else "fork")
    with ctx.Pool(workers,maxtasksperchild=1) as pool:
        for k,res in enumerate(pool.imap_unordered(_solve_reference_chunk,tasks),1):
            st,cs,gs,dr,ov,cl=res;en=st+len(cs);c_store[st:en]=cs;g_store[st:en]=gs;daily_rows.extend(dr);overlap_rows.extend(ov);cleaned.extend(cl);print(f"reference chunks {k}/{len(tasks)}",flush=True)
    D,S,A,P=c_store.shape
    period_df=pd.DataFrame({"date":np.repeat(dates.to_numpy(),S*A*P),"schedule":np.tile(np.repeat(np.asarray(labels,dtype=object),A*P),D),"asset":np.tile(np.repeat(np.asarray([a.name for a in ASSETS],dtype=object),P),D*S),"settlement_period":np.tile(np.arange(1,P+1),D*S*A),"charge_mw":c_store.reshape(-1),"discharge_mw":g_store.reshape(-1)})
    period_path=out/"controller_period_schedules_1081.csv.gz";period_df.to_csv(period_path,index=False,compression=GZIP_COMPRESSION)
    daily=pd.DataFrame(daily_rows).sort_values(["date","schedule"]);daily.to_csv(out/"controller_daily_results_1081.csv",index=False)
    summary=daily.groupby("schedule",as_index=False).agg(days=("date","nunique"),gross_gbp=("gross_gbp","sum"),per_unit_score_gbp=("per_unit_score_gbp","sum"),shared_score_gbp=("shared_score_gbp","sum"),throughput_mwh_day=("throughput_mwh","mean"),per_unit_excess_mwh_day=("per_unit_excess_mwh","mean"),shared_excess_mwh_day=("shared_excess_mwh","mean"),shared_charge_excess_mwh_day=("shared_charge_excess_mwh","mean"),shared_discharge_excess_mwh_day=("shared_discharge_excess_mwh","mean"));summary["gross_gbp_million"]=summary.gross_gbp/1e6;summary["per_unit_score_gbp_million"]=summary.per_unit_score_gbp/1e6;summary["shared_score_gbp_million"]=summary.shared_score_gbp/1e6;summary.to_csv(out/"controller_summary_1081.csv",index=False)
    overlaps=pd.DataFrame(overlap_rows);overlaps.to_csv(out/"simultaneous_charge_discharge_periods.csv",index=False)
    if len(overlaps): overlaps.groupby("schedule",as_index=False).agg(flagged_asset_periods=("overlap_mwh","size"),flagged_days=("date","nunique"),overlap_mwh=("overlap_mwh","sum"),negative_forecast_price_periods=("negative_forecast_price","sum"),negative_realized_price_periods=("negative_realized_price","sum"),absolute_price_value_bound_gbp=("absolute_price_value_bound_gbp","sum")).to_csv(out/"simultaneous_operation_summary.csv",index=False)
    clean=pd.DataFrame(cleaned);clean.groupby("schedule",as_index=False).agg(gross_gbp=("gross_gbp","sum"),per_unit_score_gbp=("per_unit_score_gbp","sum"),shared_score_gbp=("shared_score_gbp","sum"),throughput_mwh_day=("throughput_mwh","mean")).to_csv(out/"overlap_removed_summary.csv",index=False)
    return {**inputs,"features":features,"base_forecast":fc,"daily":daily,"periods":period_df,"summary":summary}


def _solve_grid_chunk(payload):
    start,date_values,fmat,pmat,vmat,svmat,pnc,png,rho=payload
    hard=SharedLP("hard");priced=SharedLP("priced");rows=[]
    for j,datev in enumerate(date_values):
        date=pd.Timestamp(datev);f=fmat[j];rp=pmat[j];rv=vmat[j];sv=svmat[j];hc=hard.solve(f,shared_cap(sv,rho),BASE_PHI)
        for phi in PHI_GRID:
            schedules={"PN declaration":(pnc[j],png[j]),"Shared hard":hc,"Shared priced":priced.solve(f,shared_cap(sv,rho),phi)}
            for label,(c,g) in schedules.items():
                m=evaluate(c,g,rp,rv,rho,phi);rows.append({"date":date,"rho":rho,"phi_gbp_mwh":phi,"schedule":label,"shared_score_gbp":m["shared_score_gbp"],"gross_gbp":m["gross_gbp"],"shared_excess_mwh":m["shared_excess_mwh"]})
    return start,rows


def _grid_worker_to_csv(root: Path, out: Path, rho: float, start: int, end: int, target: Path) -> None:
    ref=load_reference_readonly(root,out);dates=ref["matched"][start:end];fc=ref["base_forecast"]
    fmat=fc.pivot(index="date",columns="period",values="forecast_price").loc[dates,range(1,49)].to_numpy(float)
    pmat=ref["price"].loc[dates,PERIOD_COLS].to_numpy(float);vmat=ref["volume"].loc[dates,PERIOD_COLS].to_numpy(float);svmat=ref["sched_volume"].loc[dates,PERIOD_COLS].to_numpy(float)
    pnc=np.zeros((len(dates),len(ASSETS),PERIODS));png=np.zeros_like(pnc)
    for i,d in enumerate(dates):pnc[i],png[i]=pn_arrays(ref["pn"],d)
    _,rows=_solve_grid_chunk((start,dates.to_numpy(),fmat,pmat,vmat,svmat,pnc,png,rho))
    pd.DataFrame(rows).to_csv(target,index=False)


def rebuild_grid(root: Path,out: Path,reference: dict | None=None) -> dict:
    """Rebuild the 25-cell aggregate schedule surface."""
    import subprocess
    ref=reference or load_or_rebuild_reference(root,out);dates=ref["matched"]
    parts=out/"grid_parts";parts.mkdir(exist_ok=True)
    chunk=400;part_paths=[];counter=0;total=len(RHO_GRID)*math.ceil(len(dates)/chunk)
    env=os.environ.copy();env["BESS_SUPPLEMENT_ROOT"]=str(root);env["BESS_OUTPUT_ROOT"]=str(out);env["BESS_REUSE_FORECAST"]="1";env["OMP_NUM_THREADS"]="1";env["OPENBLAS_NUM_THREADS"]="1";env["MKL_NUM_THREADS"]="1";env["NUMEXPR_NUM_THREADS"]="1"
    for rho in RHO_GRID:
        for st in range(0,len(dates),chunk):
            en=min(st+chunk,len(dates));target=parts/f"rho_{rho:g}_{st}_{en}.csv";part_paths.append(target)
            cmd=[sys.executable,str(Path(__file__).resolve()),"--grid-worker","--grid-rho",str(rho),"--grid-start",str(st),"--grid-end",str(en),"--grid-output",str(target)]
            subprocess.run(cmd,check=True,env=env,stdout=subprocess.DEVNULL);counter+=1;print(f"grid parts {counter}/{total}",flush=True)
    daily=pd.concat([pd.read_csv(p,parse_dates=["date"]) for p in part_paths],ignore_index=True).sort_values(["rho","phi_gbp_mwh","date","schedule"])
    daily.to_csv(out/"parameter_surface_daily_results.csv.gz",index=False,compression=GZIP_COMPRESSION)
    agg=daily.groupby(["rho","phi_gbp_mwh","schedule"],as_index=False).shared_score_gbp.sum()
    scores=agg.pivot(index=["rho","phi_gbp_mwh"],columns="schedule",values="shared_score_gbp").reset_index().rename(columns={"PN declaration":"pn_score_gbp","Shared hard":"shared_hard_score_gbp","Shared priced":"shared_priced_score_gbp"}).sort_values(["rho","phi_gbp_mwh"])
    scores["hard_priced_abs_difference_gbp"]=(scores["shared_hard_score_gbp"]-scores["shared_priced_score_gbp"]).abs()
    for path in part_paths:
        path.unlink(missing_ok=True)
    parts.rmdir()
    return {"scores":scores,"daily":daily}


def rebuild_forecasts(root:Path,out:Path,reference:dict|None=None)->dict:
    ref=reference or load_or_rebuild_reference(root,out);features,dates=ref["features"],ref["matched"]
    rows=[]
    for look in RIDGE_LOOKBACKS:
        for alpha in RIDGE_ALPHAS:
            fc=rolling_ridge(features,dates,look,alpha);m=forecast_metrics(fc);rows.append({"training_days":look,"alpha":alpha,**m});print(f"forecast grid {look} {alpha}",flush=True)
    grid=pd.DataFrame(rows);grid["rmse_rank"]=grid.period_rmse_gbp_mwh.rank(method="min").astype(int);grid["mae_rank"]=grid.period_mae_gbp_mwh.rank(method="min").astype(int);grid.to_csv(out/"forecast_grid_current_20_cells.csv",index=False)
    best=grid.sort_values("period_rmse_gbp_mwh").iloc[0]
    variants={"base_ridge":ref["base_forecast"],"best_ridge":rolling_ridge(features,dates,int(best.training_days),float(best.alpha)),"gradient_boosting":rolling_gbm(features,dates)}
    price,volume,svol=ref["price"],ref["volume"],ref["sched_volume"];solver=SharedLP("priced")
    sens=[]
    for name,fc in variants.items():
        fm=forecast_metrics(fc);fmap=forecast_map(fc);drows=[]
        for date in dates:
            f=fmap[date];rp=price.loc[date,PERIOD_COLS].to_numpy(float);rv=volume.loc[date,PERIOD_COLS].to_numpy(float);sv=svol.loc[date,PERIOD_COLS].to_numpy(float);c,g=solver.solve(f,shared_cap(sv,BASE_RHO),BASE_PHI);m=evaluate(c,g,rp,rv,BASE_RHO,BASE_PHI);drows.append({"date":date,"forecast_model":name,**m})
        dd=pd.DataFrame(drows)
        if name in {"best_ridge","gradient_boosting"}:
            dd.to_csv(out/f"shared_priced_{name}_daily.csv",index=False)
        sens.append({"forecast_model":name,**fm,"shared_priced_score_gbp":float(dd.shared_score_gbp.sum()),"shared_priced_score_gbp_million":float(dd.shared_score_gbp.sum()/1e6),"throughput_mwh_day":float(dd.throughput_mwh.mean()),"shared_excess_mwh_day":float(dd.shared_excess_mwh.mean())})
    sensitivity=pd.DataFrame(sens);base_score=float(sensitivity.loc[sensitivity.forecast_model=="base_ridge","shared_priced_score_gbp"].iloc[0]);sensitivity["score_change_vs_base_percent"]=100*(sensitivity.shared_priced_score_gbp/base_score-1);sensitivity.to_csv(out/"forecast_learner_shared_sensitivity.csv",index=False)
    return {"grid":grid,"sensitivity":sensitivity}


def rebuild_operational_summary(root:Path,out:Path,inputs:dict|None=None)->pd.DataFrame:
    """Rebuild the deposited descriptive PN, B1610 and BOALF summary."""
    inputs=inputs or build_current_inputs(root,out);dates=inputs["matched"];price=inputs["price"];pn=inputs["pn"]
    b=pd.read_csv(root/"inputs"/"current_snapshot"/"current_latest_b1610_asset_period_panel.csv",parse_dates=["settlementDate"]);b=b[b.settlementDate.isin(dates)&b.settlementPeriod.between(1,48)]
    pl=price.loc[dates,PERIOD_COLS].rename_axis("date").reset_index().melt(id_vars="date",var_name="settlementPeriod",value_name="price");pl.settlementPeriod=pl.settlementPeriod.astype(int);pl=pl.rename(columns={"date":"settlementDate"})
    p=pn[pn.settlementDate.isin(dates)&pn.settlementPeriod.between(1,48)][["asset","settlementDate","settlementPeriod","pnMWh"]].merge(pl,on=["settlementDate","settlementPeriod"])
    bb=b[["asset","settlementDate","settlementPeriod","quantity"]].merge(pl,on=["settlementDate","settlementPeriod"])
    bo=pd.read_csv(root/"inputs"/"extensions"/"boalf_period_level_clean_2023_2025.csv",parse_dates=["settlementDate"]);bo["asset"]=bo.asset.replace({"Pillswood 1 Battery Storage":"Pillswood Battery Storage","Pillswood 2 Battery Storage":"Pillswood Battery Storage"});bo=bo[bo.asset.isin([a.name for a in ASSETS])&bo.settlementDate.isin(dates)&bo.settlementPeriod.between(1,48)][["asset","settlementDate","settlementPeriod","boalfMWh"]].groupby(["asset","settlementDate","settlementPeriod"],as_index=False).boalfMWh.sum()
    m=p[["asset","settlementDate","settlementPeriod","pnMWh","price"]].merge(bb[["asset","settlementDate","settlementPeriod","quantity"]],on=["asset","settlementDate","settlementPeriod"],how="inner").merge(bo,on=["asset","settlementDate","settlementPeriod"],how="left");m.boalfMWh=m.boalfMWh.fillna(0.0);m["deviation_mwh"]=m.quantity-m.pnMWh;m["aligned_mwh"]=np.where(np.sign(m.deviation_mwh)==np.sign(m.boalfMWh),np.minimum(m.deviation_mwh.abs(),m.boalfMWh.abs()),0.0);m["pn_gross_mark_gbp"]=m.pnMWh*m.price;m["b1610_gross_mark_gbp"]=m.quantity*m.price
    rows=[]
    for asset,g in m.groupby("asset"):
        absdev=float(g.deviation_mwh.abs().sum());aligned=float(g.aligned_mwh.sum());ndays=int(g.settlementDate.nunique());rows.append({"asset":asset,"days":ndays,"pn_gross_mark_gbp":float(g.pn_gross_mark_gbp.sum()),"b1610_gross_mark_gbp":float(g.b1610_gross_mark_gbp.sum()),"absolute_deviation_mwh_day":absdev/ndays,"boalf_aligned_mwh_day":aligned/ndays,"boalf_aligned_share_percent":100*aligned/absdev,"net_boalf_mwh":float(g.boalfMWh.sum()),"negative_b1610_row_share":float(np.mean(g.quantity<0))})
    absdev=float(m.deviation_mwh.abs().sum());aligned=float(m.aligned_mwh.sum());ndays=int(m.settlementDate.nunique());rows.append({"asset":"Portfolio","days":ndays,"pn_gross_mark_gbp":float(m.pn_gross_mark_gbp.sum()),"b1610_gross_mark_gbp":float(m.b1610_gross_mark_gbp.sum()),"absolute_deviation_mwh_day":absdev/ndays,"boalf_aligned_mwh_day":aligned/ndays,"boalf_aligned_share_percent":100*aligned/absdev,"net_boalf_mwh":float(m.boalfMWh.sum()),"negative_b1610_row_share":float(np.mean(m.quantity<0))})
    summary=pd.DataFrame(rows);summary.to_csv(out/"declaration_metering_boalf_summary.csv",index=False);return summary


def load_or_rebuild_reference(root:Path,out:Path)->dict:
    req=[out/"controller_daily_results_1081.csv",out/"controller_period_schedules_1081.csv.gz",out/"forecast_base_ridge_period.csv",out/"feature_panel_current.csv.gz",out/"matched_dates_1081.csv"]
    if not all(p.exists() for p in req): return rebuild_reference(root,out)
    inputs=build_current_inputs(root,out);return {**inputs,"features":pd.read_csv(out/"feature_panel_current.csv.gz",parse_dates=["date"]),"base_forecast":pd.read_csv(out/"forecast_base_ridge_period.csv",parse_dates=["date"]),"daily":pd.read_csv(out/"controller_daily_results_1081.csv",parse_dates=["date"]),"periods":None,"summary":pd.read_csv(out/"controller_summary_1081.csv")}


def load_reference_readonly(root:Path,out:Path)->dict:
    """Load the stored reference layer without changing package files."""
    price=read_pivot(out/"mid_price_complete_2022_2025.csv");volume=read_pivot(out/"mid_volume_complete_2022_2025.csv");sched_volume=read_pivot(out/"mid_volume_trailing28_median.csv")
    matched=pd.DatetimeIndex(pd.read_csv(out/"matched_dates_1081.csv",parse_dates=["date"])["date"]);pn=pd.read_csv(root/"inputs"/"current_snapshot"/"current_latest_pn_asset_period_panel.csv",parse_dates=["settlementDate"]);panel=json.loads((out/"panel_construction_1081.json").read_text(encoding="utf-8"))
    return {"price":price,"volume":volume,"sched_volume":sched_volume,"matched":matched,"pn":pn,"panel":panel,"features":pd.read_csv(out/"feature_panel_current.csv.gz",parse_dates=["date"]),"base_forecast":pd.read_csv(out/"forecast_base_ridge_period.csv",parse_dates=["date"]),"daily":pd.read_csv(out/"controller_daily_results_1081.csv",parse_dates=["date"]),"periods":None,"summary":pd.read_csv(out/"controller_summary_1081.csv")}


def validate_stored(root:Path,out:Path,tol:float=1e-6)->dict:
    """Validate every retained result layer used by the article or supplement."""
    required=[
        "controller_daily_results_1081.csv","controller_period_schedules_1081.csv.gz","controller_summary_1081.csv",
        "parameter_surface_daily_results.csv.gz",
        "aggregation_difference_bootstrap_summary.csv","aggregation_difference_bootstrap_replicates.csv.gz",
        "annual_reference_aggregation_decomposition.csv","forecast_learner_shared_sensitivity.csv",
        "shared_priced_best_ridge_daily.csv","shared_priced_gradient_boosting_daily.csv",
        "simultaneous_charge_discharge_periods.csv","simultaneous_operation_summary.csv","overlap_removed_summary.csv",
        "declaration_metering_boalf_summary.csv","panel_construction_1081.json",
        "cap_assignment_surface_daily.csv.gz","cap_assignment_surface_25_cells.csv",
        "aggregate_cap_counterfactual_daily.csv.gz","aggregate_cap_counterfactual_summary.csv",
    ]
    missing=[name for name in required if not (out/name).exists()]
    if missing: raise RuntimeError(f"Missing stored outputs: {missing}")
    checks={};errors=[]

    daily=pd.read_csv(out/"controller_daily_results_1081.csv",parse_dates=["date"]);summary=pd.read_csv(out/"controller_summary_1081.csv")
    calc=daily.groupby("schedule",as_index=False).agg(gross_gbp=("gross_gbp","sum"),per_unit_score_gbp=("per_unit_score_gbp","sum"),shared_score_gbp=("shared_score_gbp","sum"),throughput_mwh_day=("throughput_mwh","mean"),per_unit_excess_mwh_day=("per_unit_excess_mwh","mean"),shared_excess_mwh_day=("shared_excess_mwh","mean"))
    merged=calc.merge(summary,on="schedule",suffixes=("_calc","_stored"));fields=["gross_gbp","per_unit_score_gbp","shared_score_gbp","throughput_mwh_day","per_unit_excess_mwh_day","shared_excess_mwh_day"]
    core_error=max(float(np.max(np.abs(merged[f+"_calc"]-merged[f+"_stored"]))) for f in fields);checks["reference_daily_to_summary_max_abs"]=core_error;errors.append(core_error)

    period=pd.read_csv(out/"controller_period_schedules_1081.csv.gz",parse_dates=["date"]);expected=1081*6*3*48;bad_groups=int((period.groupby(["date","schedule","asset"]).settlement_period.nunique()!=48).sum());period_error=float(abs(len(period)-expected)+bad_groups);checks["period_schedule"]={"rows":int(len(period)),"expected_rows":expected,"groups_not_equal_48":bad_groups};errors.append(period_error)

    surface_daily=pd.read_csv(out/"parameter_surface_daily_results.csv.gz")

    bs=pd.read_csv(out/"aggregation_difference_bootstrap_summary.csv");reps=pd.read_csv(out/"aggregation_difference_bootstrap_replicates.csv.gz");bootstrap_errors=[]
    for row in bs.itertuples(index=False):
        values=reps.loc[reps.block_days==row.block_days,"mean_kGBP_day"].to_numpy(float);lo,hi=np.quantile(values,[.025,.975]);bootstrap_errors.extend([abs(float(lo)-float(row.ci_low_kGBP_day)),abs(float(hi)-float(row.ci_high_kGBP_day))])
    bootstrap_error=float(max(bootstrap_errors) if bootstrap_errors else 0.0);checks["bootstrap_summary_max_abs_kGBP_day"]=bootstrap_error;errors.append(bootstrap_error*1000.0)

    forecast_summary=pd.read_csv(out/"forecast_learner_shared_sensitivity.csv");score_lookup={"base_ridge":float(daily.loc[daily.schedule=="Shared priced","shared_score_gbp"].sum()),"best_ridge":float(pd.read_csv(out/"shared_priced_best_ridge_daily.csv").shared_score_gbp.sum()),"gradient_boosting":float(pd.read_csv(out/"shared_priced_gradient_boosting_daily.csv").shared_score_gbp.sum())};forecast_error=max(abs(score_lookup[name]-float(forecast_summary.loc[forecast_summary.forecast_model==name,"shared_priced_score_gbp"].iloc[0])) for name in score_lookup);checks["forecast_daily_to_summary_max_abs_gbp"]=forecast_error;errors.append(forecast_error)

    sim_period=pd.read_csv(out/"simultaneous_charge_discharge_periods.csv",parse_dates=["date"]);sim_summary=pd.read_csv(out/"simultaneous_operation_summary.csv");sim_errors=[]
    for schedule,group in sim_period.groupby("schedule"):
        row=sim_summary.loc[sim_summary.schedule==schedule].iloc[0];sim_errors.extend([abs(len(group)-int(row.flagged_asset_periods)),abs(group.date.nunique()-int(row.flagged_days)),abs(float(group.overlap_mwh.sum())-float(row.overlap_mwh)),abs(float(group.absolute_price_value_bound_gbp.sum())-float(row.absolute_price_value_bound_gbp))])
    sim_error=float(max(sim_errors) if sim_errors else 0.0);checks["simultaneous_operation_max_abs"]=sim_error;errors.append(sim_error)

    pu=pd.read_csv(out/"cap_assignment_surface_daily.csv.gz",parse_dates=["date"]);pu=pu[(pu.rho==BASE_RHO)&(pu.phi_gbp_mwh==BASE_PHI)][["date","per_unit_priced_per_unit_score_gbp","per_unit_priced_shared_score_gbp"]]
    sh=surface_daily[(surface_daily.rho==BASE_RHO)&(surface_daily.phi_gbp_mwh==BASE_PHI)&(surface_daily.schedule=="Shared priced")][["date","shared_score_gbp"]].copy();sh["date"]=pd.to_datetime(sh["date"])
    annual=pu.merge(sh,on="date",validate="one_to_one");annual["year"]=annual.date.dt.year;annual["optimised_wedge_gbp"]=annual.per_unit_priced_per_unit_score_gbp-annual.shared_score_gbp;annual["audit_gap_gbp"]=annual.per_unit_priced_per_unit_score_gbp-annual.per_unit_priced_shared_score_gbp;annual["recovery_gbp"]=annual.shared_score_gbp-annual.per_unit_priced_shared_score_gbp;annual_calc=annual.groupby("year",as_index=False)[["optimised_wedge_gbp","audit_gap_gbp","recovery_gbp"]].sum();annual_stored=pd.read_csv(out/"annual_reference_aggregation_decomposition.csv");am=annual_calc.merge(annual_stored,on="year",suffixes=("_calc","_stored"));annual_error=max(float(np.max(np.abs(am[f+"_calc"]-am[f+"_stored"]))) for f in ["optimised_wedge_gbp","audit_gap_gbp","recovery_gbp"]);checks["annual_decomposition_max_abs_gbp"]=annual_error;errors.append(annual_error)

    ref=load_reference_readonly(root,out);fmap=forecast_map(ref["base_forecast"]);solver=SharedLP("priced");sentinel=[]
    for date in [ref["matched"][0],ref["matched"][len(ref["matched"])//2],ref["matched"][-1]]:
        f=fmap[date];sv=ref["sched_volume"].loc[date,PERIOD_COLS].to_numpy(float);c,g=solver.solve(f,shared_cap(sv,BASE_RHO),BASE_PHI);stored=period[(period.date==date)&(period.schedule=="Shared priced")];cs=np.vstack([stored[stored.asset==a.name].sort_values("settlement_period").charge_mw.to_numpy(float) for a in ASSETS]);gs=np.vstack([stored[stored.asset==a.name].sort_values("settlement_period").discharge_mw.to_numpy(float) for a in ASSETS]);sentinel.append({"date":str(date.date()),"max_abs_charge_mw":float(np.max(np.abs(c-cs))),"max_abs_discharge_mw":float(np.max(np.abs(g-gs)))})
    sentinel_error=max(max(x["max_abs_charge_mw"],x["max_abs_discharge_mw"]) for x in sentinel);checks["sentinel_solve_errors"]=sentinel;errors.append(sentinel_error)

    panel=json.loads((out/"panel_construction_1081.json").read_text(encoding="utf-8"));panel_error=float(abs(int(daily.date.nunique())-1081)+abs(int(panel.get("matched_days",0))-1081));checks["panel"]={"stored":panel,"daily_days":int(daily.date.nunique())};errors.append(panel_error)
    operational=pd.read_csv(out/"declaration_metering_boalf_summary.csv");expected_assets={a.name for a in ASSETS}|{"Portfolio"};operational_error=0.0 if set(operational.asset)==expected_assets and (operational.days==1081).all() else 1.0;checks["operational_summary_rows"]=operational.to_dict(orient="records");errors.append(operational_error)

    from run_scope_extensions import validate_stored as validate_scope
    scope=validate_scope(root,out,tol=tol);scope_error=max(float(scope["surface_daily_to_summary_max_abs_gbp"]),float(scope["shared_surface_to_cap_summary_max_abs_gbp"]),float(scope["allowance_daily_to_summary_max_abs_gbp"]),float(scope["allowance_baseline_to_reference_max_abs_gbp"]));checks["scope_extensions"]=scope;errors.append(scope_error)

    max_error=float(max(errors) if errors else 0.0);report={"status":"PASS" if max_error<=tol else "FAIL","tolerance":tol,"max_error":max_error,"checks":checks}
    if report["status"]!="PASS": raise RuntimeError(report)
    print(json.dumps(report,indent=2));return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--rebuild-reference",action="store_true");ap.add_argument("--reoptimise-grid",action="store_true");ap.add_argument("--rebuild-forecasts",action="store_true");ap.add_argument("--rebuild-operational",action="store_true");ap.add_argument("--rebuild-all",action="store_true");ap.add_argument("--validate-stored",action="store_true");ap.add_argument("--grid-worker",action="store_true",help=argparse.SUPPRESS);ap.add_argument("--grid-rho",type=float,help=argparse.SUPPRESS);ap.add_argument("--grid-start",type=int,help=argparse.SUPPRESS);ap.add_argument("--grid-end",type=int,help=argparse.SUPPRESS);ap.add_argument("--grid-output",help=argparse.SUPPRESS);args=ap.parse_args()
    root=detect_root();out=output_root(root);ref=None
    if args.grid_worker:
        _grid_worker_to_csv(root,out,args.grid_rho,args.grid_start,args.grid_end,Path(args.grid_output));return
    if args.rebuild_all or args.rebuild_reference: ref=rebuild_reference(root,out)
    if args.rebuild_all or args.reoptimise_grid: ref=ref or load_or_rebuild_reference(root,out);rebuild_grid(root,out,ref)
    if args.rebuild_all or args.rebuild_forecasts: ref=ref or load_or_rebuild_reference(root,out);rebuild_forecasts(root,out,ref)
    if args.rebuild_all or args.rebuild_operational: ref=ref or load_or_rebuild_reference(root,out);rebuild_operational_summary(root,out,ref)
    if args.validate_stored: validate_stored(root,out)
    if not any(vars(args).values()): ap.print_help()


if __name__=="__main__": main()
