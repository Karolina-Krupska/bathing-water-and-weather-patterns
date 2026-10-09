#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Step 5 — BWQ exceedance "risk score" models + operational burden metrics (SWEW, IE>=63)

This version keeps all existing models A–G and adds:
  H_soil_API_only
  I_wind_UV_SSRD_t2m_only
  J_wind_UV_SSRD_t2m_plus_soil_API

NEW OUTPUT (when --sim-drivers-file is provided):
  A NEW folder is created (timestamped) named like:
    "probabilities for models A - J mastersheet_YYYYMMDD_HHMMSS"
  Inside it, the script writes a master daily file that contains the simulated drivers + probabilities:
    - sim_with_predictions.csv.gz  (drivers + p_* + pcal_*)
    - probs_only.csv.gz            (keys + p_* + pcal_* only)

Nothing is overwritten: all outputs go into NEW timestamped folders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional dependency used in your Q3 scripts (for NGR -> lat/lon).
try:
    from pyproj import Transformer  # type: ignore
except Exception:  # pragma: no cover
    Transformer = None  # type: ignore

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# Constants / configuration
# -----------------------------

DEFAULT_SEED = 42
DEFAULT_EXCEED_THRESHOLD = 63.0

DEFAULT_ANALYSIS_MONTHS = [5, 6, 7, 8, 9]
DEFAULT_REGION = "SWEW"
DEFAULT_YEAR_MIN = 2012
DEFAULT_YEAR_MAX = 2024

# User-specific default paths for this Step 5 run.
# Outputs are still written into NEW timestamped folders, so old outputs are not overwritten.
DEFAULT_BASE_DIR = Path(r"C:\Users\earth\OneDrive - University of Reading\PHD 2\Chapters\Paper 2\Codes 2\Analytical procedure\Step 5 Models A-J")
DEFAULT_RAINFALL_FILE = Path(r"C:\Users\earth\OneDrive - University of Reading\PHD 2\Chapters\Paper 2\Codes 2\Analytical procedure\Step 4 Historical daily package resampling\Daily rainfall metrix_AprSep_2012_2024.xlsx")
DEFAULT_PACKAGES_FILE = Path(r"C:\Users\earth\OneDrive - University of Reading\PHD 2\Chapters\Paper 2\Codes 2\Analytical procedure\Step 4 Historical daily package resampling\S4pkg_AprRain_2012_2024_n5000_20260518_155820\historical_packages.csv")
DEFAULT_SIM_DRIVERS_FILE = Path(r"C:\Users\earth\OneDrive - University of Reading\PHD 2\Chapters\Paper 2\Codes 2\Analytical procedure\Step 4 Historical daily package resampling\S4pkg_AprRain_2012_2024_n5000_20260518_155820\simulated_storyline_drivers_daily.csv.gz")

# 30-regime -> 4-group mapping
REGIME_TO_GROUP_4: Dict[int, str] = {
    1: "Cyclonic Atlantic (frontal)",
    2: "Showery maritime / unsettled",
    3: "Convective extremes",
    4: "Showery maritime / unsettled",
    5: "Cyclonic Atlantic (frontal)",
    6: "Convective extremes",
    7: "Cyclonic Atlantic (frontal)",
    8: "Cyclonic Atlantic (frontal)",
    9: "Convective extremes",
    10: "Convective extremes",
    11: "Cyclonic Atlantic (frontal)",
    12: "Convective extremes",
    13: "Settled anticyclonic quiet",
    14: "Showery maritime / unsettled",
    15: "Showery maritime / unsettled",
    16: "Convective extremes",
    17: "Convective extremes",
    18: "Convective extremes",
    19: "Cyclonic Atlantic (frontal)",
    20: "Showery maritime / unsettled",
    21: "Cyclonic Atlantic (frontal)",
    22: "Cyclonic Atlantic (frontal)",
    23: "Settled anticyclonic quiet",
    24: "Cyclonic Atlantic (frontal)",
    25: "Settled anticyclonic quiet",
    26: "Showery maritime / unsettled",
    27: "Convective extremes",
    28: "Cyclonic Atlantic (frontal)",
    29: "Cyclonic Atlantic (frontal)",
    30: "Cyclonic Atlantic (frontal)",
}

GROUP_ORDER = [
    "Cyclonic Atlantic (frontal)",
    "Showery maritime / unsettled",
    "Convective extremes",
    "Settled anticyclonic quiet",
]

# SWEW box (lat_min, lat_max, lon_min, lon_max)
SWEW_BOX = (50.1, 51.2, -4.1, -2.6)


# -----------------------------
# Small utilities
# -----------------------------

def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_first_existing(base_dir: Path, candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        p = base_dir / c
        if p.exists():
            return p
    return None


def detect_latest_matching(base_dir: Path, pattern: str) -> Optional[Path]:
    matches = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_micro_value(x) -> float:
    """Convert EA microbiology values to float. Handles strings like '<10', 'NaN', etc."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    s = s.replace(",", "")
    if s.startswith("<") or s.startswith(">"):
        s = s[1:].strip()
    try:
        return float(s)
    except Exception:
        return np.nan


# -----------------------------
# NGR -> lat/lon (optional)
# -----------------------------

_OSGB36_to_WGS84 = None
if Transformer is not None:
    _OSGB36_to_WGS84 = Transformer.from_crs("epsg:27700", "epsg:4326", always_xy=True)


def ngr_to_en(ngr: str) -> Tuple[float, float]:
    """Convert NGR string to OSGB36 Easting/Northing (meters). Minimal implementation."""
    if pd.isna(ngr):
        return (np.nan, np.nan)
    s = str(ngr).strip().replace(" ", "").upper()
    if len(s) < 4:
        return (np.nan, np.nan)

    grid = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no I
    l1, l2 = s[0], s[1]
    if l1 not in grid or l2 not in grid:
        return (np.nan, np.nan)

    i1, i2 = grid.index(l1), grid.index(l2)
    e100km = ((i1 - 2) % 5) * 5 + (i2 % 5)
    n100km = (19 - (i1 // 5) * 5) - (i2 // 5)

    digits = s[2:]
    if not digits.isdigit():
        return (np.nan, np.nan)

    half = len(digits) // 2
    e = e100km * 100000 + int(digits[:half].ljust(5, "0"))
    n = n100km * 100000 + int(digits[half:].ljust(5, "0"))
    return float(e), float(n)


def ngr_to_latlon(ngr: str) -> Tuple[float, float]:
    if _OSGB36_to_WGS84 is None:
        return (np.nan, np.nan)
    e, n = ngr_to_en(ngr)
    if np.isnan(e) or np.isnan(n):
        return (np.nan, np.nan)
    lon, lat = _OSGB36_to_WGS84.transform(e, n)
    return float(lon), float(lat)


def in_swew_box(lat: float, lon: float) -> bool:
    lat_min, lat_max, lon_min, lon_max = SWEW_BOX
    return (lat_min <= lat <= lat_max) and (lon_min <= lon <= lon_max)


# -----------------------------
# Data loading: rainfall + packages
# -----------------------------

def load_rainfall_metrics(rain_path: Path, region: str) -> pd.DataFrame:
    """Daily rainfall metrics table (e.g., 'Daily rainfall metrix.xlsx')."""
    df = pd.read_excel(rain_path)
    df = normalise_columns(df)
    if "date" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])

    if "region" in df.columns:
        df = df[df["region"].astype(str).str.upper() == region.upper()].copy()

    rename = {}
    if "daily_depth_mm" in df.columns:
        rename["daily_depth_mm"] = "depth_mm"
    if "rain_mm" in df.columns:
        rename["rain_mm"] = "depth_mm"
    if "q95_mmph" in df.columns:
        rename["q95_mmph"] = "intensity_q95_mmph"
    if "mean_rate_mmph" in df.columns:
        rename["mean_rate_mmph"] = "intensity_mean_mmph"
    df = df.rename(columns=rename)

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    if "regime" not in df.columns:
        raise ValueError(f"Rainfall file missing 'regime' column: {rain_path}")

    df["family"] = df["regime"].map(REGIME_TO_GROUP_4).astype("category")
    return df


def load_historical_packages(packages_path: Path) -> pd.DataFrame:
    """Step 4 output: historical_packages.csv (daily drivers with WP family)."""
    df = pd.read_csv(packages_path)
    df = normalise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    if "family" in df.columns:
        df["family"] = df["family"].astype("category")
    return df


# -----------------------------
# Data loading: BW chemistry
# -----------------------------

def load_bw_chemistry(files: List[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_excel(p)
        df = normalise_columns(df)

        rename = {
            "SMPT_SHORT_NAME": "site_name",
            "SMPT_USER_REFERENCE": "site_id",
            "Date": "date",
            "Time": "time",
            "2348 E_coli C-MF (NO/100ml)": "ec_raw",
            "2348 E_coli C-MF (NO/100mL)": "ec_raw",
            "3723 IE Conf (CFU/0_1l)": "ie_raw",
            "3723 IE Conf (CFU/0.1l)": "ie_raw",
            "3723 IE Conf (CFU/0_1L)": "ie_raw",
            "NGR": "ngr",
        }
        for k, v in rename.items():
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})

        req = ["site_name", "date", "ie_raw"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(
                f"BW file {p.name} is missing columns {missing}. "
                f"Columns found: {list(df.columns)}"
            )

        if "site_id" not in df.columns:
            df["site_id"] = df["site_name"].astype(str)

        df["date"] = pd.to_datetime(df["date"])
        df["ie"] = df["ie_raw"].apply(clean_micro_value)
        df["ec"] = df["ec_raw"].apply(clean_micro_value) if "ec_raw" in df.columns else np.nan

        if "lat" in df.columns and "lon" in df.columns:
            df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
            df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        else:
            if "ngr" in df.columns and Transformer is not None:
                ll = df["ngr"].apply(lambda g: pd.Series(ngr_to_latlon(g), index=["lon", "lat"]))
                df = pd.concat([df, ll], axis=1)
            else:
                warnings.warn(
                    "BW chemistry file has no (lat, lon) and pyproj/NGR conversion unavailable. "
                    "Proceeding WITHOUT SWEW spatial filter; ensure your file already contains only SWEW sites."
                )
                df["lon"] = np.nan
                df["lat"] = np.nan

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def make_exceedance_dataset(
    bw: pd.DataFrame,
    threshold: float,
    months: List[int],
    target_level: str = "site_day",
) -> pd.DataFrame:
    """Return Y dataset with columns: date, year, month, site_id (optional), y_exceed."""
    bw = bw.copy()
    bw["year"] = bw["date"].dt.year
    bw["month"] = bw["date"].dt.month
    bw = bw[bw["month"].isin(months)].copy()

    if bw["lat"].notna().any() and bw["lon"].notna().any():
        mask = bw.apply(lambda r: in_swew_box(r["lat"], r["lon"]), axis=1)
        bw = bw[mask].copy()

    bw["y_exceed"] = (bw["ie"] >= threshold).astype(int)

    if target_level == "site_day":
        g = bw.groupby(["site_id", "date"], as_index=False)["y_exceed"].max()
        g["year"] = g["date"].dt.year
        g["month"] = g["date"].dt.month
        return g
    elif target_level == "day_any":
        g = bw.groupby(["date"], as_index=False)["y_exceed"].max()
        g["year"] = g["date"].dt.year
        g["month"] = g["date"].dt.month
        return g
    else:
        raise ValueError("target_level must be 'site_day' or 'day_any'")


# -----------------------------
# Modelling helpers
# -----------------------------

@dataclass
class ModelSpec:
    name: str
    features_num: List[str]
    features_cat: List[str]


def build_pipeline(num_cols: List[str], cat_cols: List[str], seed: int) -> Pipeline:
    """Preprocess + logistic regression; robust to missing values via imputation."""
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            )
        )

    pre = ColumnTransformer(transformers, remainder="drop")
    clf = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=seed,
        class_weight="balanced",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def lo_year_out_cv_predict_calibrated(
    df: pd.DataFrame,
    y_col: str,
    year_col: str,
    model: Pipeline,
    feature_cols: List[str],
    method: str = "sigmoid",
) -> Tuple[pd.Series, pd.Series]:
    """LOYO out-of-fold predictions, returned as (raw, calibrated) with no test-year leakage."""
    preds_raw = pd.Series(index=df.index, dtype=float)
    preds_cal = pd.Series(index=df.index, dtype=float)

    years = sorted(df[year_col].dropna().unique())
    for y in years:
        train = df[df[year_col] != y]
        test = df[df[year_col] == y]
        if len(test) == 0 or len(train) == 0:
            continue

        model.fit(train[feature_cols], train[y_col])

        p_test = model.predict_proba(test[feature_cols])[:, 1]

        try:
            score_train = model.decision_function(train[feature_cols])
            score_test = model.decision_function(test[feature_cols])
        except Exception:
            p_train = model.predict_proba(train[feature_cols])[:, 1]
            eps = 1e-6
            p_train = np.clip(p_train, eps, 1 - eps)
            p_test_clip = np.clip(p_test, eps, 1 - eps)
            score_train = np.log(p_train / (1 - p_train))
            score_test = np.log(p_test_clip / (1 - p_test_clip))

        if method.lower() == "sigmoid":
            cal = LogisticRegression(max_iter=2000, solver="lbfgs")
            cal.fit(np.asarray(score_train).reshape(-1, 1), train[y_col].to_numpy())
            p_test_cal = cal.predict_proba(np.asarray(score_test).reshape(-1, 1))[:, 1]
        elif method.lower() == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(np.asarray(score_train), train[y_col].to_numpy())
            p_test_cal = cal.transform(np.asarray(score_test))
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        preds_raw.loc[test.index] = p_test
        preds_cal.loc[test.index] = p_test_cal

    return preds_raw, preds_cal


def compute_burden_metrics(p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Burden metrics from a daily p(t) sequence."""
    hi = (p >= threshold).astype(int)

    n_hi = int(hi.sum())

    runs = []
    cur = 0
    for v in hi:
        if v == 1:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
                cur = 0
    if cur > 0:
        runs.append(cur)

    n_clusters = int(len(runs))
    max_run = int(max(runs)) if runs else 0
    mean_run = float(np.mean(runs)) if runs else 0.0

    gaps = []
    in_run = False
    last_end = None
    for i, v in enumerate(hi):
        if v == 1 and not in_run:
            in_run = True
        if v == 0 and in_run:
            in_run = False
            last_end = i - 1
        if v == 1 and (last_end is not None):
            gaps.append(i - last_end - 1)
            last_end = None
    ttr_mean = float(np.mean(gaps)) if gaps else np.nan

    return {
        "high_risk_days": n_hi,
        "clusters": n_clusters,
        "max_run": max_run,
        "mean_run": mean_run,
        "mean_time_to_recovery_days": ttr_mean,
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_roc_curves(
    out_path: Path,
    y: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    *,
    label_map: Optional[Dict[str, str]] = None,
    legend_fontsize: int = 9,
    title: str = "ROC curves — SWEW (IE ≥ 63)",
) -> None:
    label_map = label_map or {}

    plt.figure(figsize=(7.5, 6))
    for name, p in pred_dict.items():
        mask = np.isfinite(p)
        if mask.sum() < 10:
            continue
        fpr, tpr, _ = roc_curve(y[mask], p[mask])
        aucv = roc_auc_score(y[mask], p[mask])
        pretty = label_map.get(name, name)
        plt.plot(fpr, tpr, label=f"{pretty} (AUC={aucv:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True, fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_calibration_deciles(
    out_path: Path,
    y: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    *,
    title: str = "Calibration (deciles) — SWEW (IE ≥ 63)",
    label_map: Optional[Dict[str, str]] = None,
    legend_fontsize: int = 9,
) -> None:
    label_map = label_map or {}

    plt.figure(figsize=(7.5, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    for name, p in pred_dict.items():
        mask = np.isfinite(p)
        if mask.sum() < 10:
            continue

        q = np.nanquantile(p[mask], np.linspace(0, 1, 11))
        q[0] = -1e9
        q[-1] = 1e9
        bin_idx = np.digitize(p[mask], q) - 1

        xs, ys = [], []
        for b in range(10):
            sel = bin_idx == b
            if sel.sum() < 5:
                continue
            xs.append(float(np.mean(p[mask][sel])))
            ys.append(float(np.mean(y[mask][sel])))

        pretty = label_map.get(name, name)
        plt.plot(xs, ys, marker="o", label=pretty)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend(loc="upper left", frameon=True, fontsize=legend_fontsize)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_depth_exceed_by_family(out_path: Path, df: pd.DataFrame, y_col: str, thr: float) -> None:
    """Simple diagnostic: exceedance fraction vs depth bins by WP family."""
    if "depth_mm" not in df.columns or "family" not in df.columns:
        return
    d = df.copy()
    d = d[np.isfinite(d["depth_mm"])].copy()
    bins = np.array([0, 0.2, 0.5, 1, 2, 3, 5, 8, 12, 20])
    d["bin"] = pd.cut(d["depth_mm"], bins=bins, include_lowest=True)
    g = d.groupby(["family", "bin"], observed=True)[y_col].agg(["mean", "count"]).reset_index()
    mids = [(b.left + b.right) / 2.0 for b in g["bin"]]
    g["mid"] = mids

    plt.figure(figsize=(7.5, 6))
    for fam in GROUP_ORDER:
        sub = g[g["family"] == fam]
        if len(sub) == 0:
            continue
        plt.plot(sub["mid"], sub["mean"], marker="o", label=fam)
    plt.xlabel("Daily rainfall depth (mm)")
    plt.ylabel(f"Exceedance fraction (IE ≥ {thr:g})")
    plt.title("Rainfall depth vs exceedance by WP family — SWEW")
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR), help="Base folder containing rainfall + BW chemistry files (and optionally Step4 outputs).")
    p.add_argument("--rainfall-file", default=str(DEFAULT_RAINFALL_FILE), help="Path to 'Daily rainfall metrix.xlsx' (optional if in --base-dir).")
    p.add_argument("--packages-file", default=str(DEFAULT_PACKAGES_FILE), help="Path to Step4 'historical_packages.csv' (optional).")
    p.add_argument("--sim-drivers-file", default=str(DEFAULT_SIM_DRIVERS_FILE), help="Optional: simulated daily drivers file to compute burden metrics AND write A–J probability mastersheet.")
    p.add_argument("--out-dir", default=None, help="Output ROOT folder. A NEW timestamped folder is always created inside this root.")
    p.add_argument("--region", default=DEFAULT_REGION, help="Region name (default SWEW).")
    p.add_argument("--analysis-months", default="5,6,7,8,9", help="Comma-separated months (default May-Sep).")
    p.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN, help="First analysis year to keep (default 2012).")
    p.add_argument("--year-max", type=int, default=DEFAULT_YEAR_MAX, help="Last analysis year to keep (default 2024).")
    p.add_argument("--exceed-threshold", type=float, default=DEFAULT_EXCEED_THRESHOLD, help="IE exceedance threshold (default 63).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (reproducibility).")
    p.add_argument("--target-level", choices=["site_day", "day_any"], default="site_day", help="Target definition: site-day (default) or any-site-per-day.")
    p.add_argument("--high-risk-threshold", type=float, default=0.5, help="Threshold for 'high-risk day' burden metrics.")
    p.add_argument("--calibration-method", dest="calibration_method", choices=["sigmoid", "isotonic"], default="sigmoid",
                   help="Probability calibration inside LOYO folds (default sigmoid).")
    p.add_argument("--save-sim-mastersheet", action="store_true", default=True,
                   help="Writes 'sim_with_predictions.csv.gz' + probs_only.csv.gz into a dedicated 'probabilities for models A - J mastersheet_*' folder. Default is True for this 2012-2024 run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = Path(args.base_dir)

    months = [int(x) for x in str(args.analysis_months).split(",") if str(x).strip()]

    # --- Detect inputs
    rain_path = Path(args.rainfall_file) if args.rainfall_file else detect_first_existing(
        base, ["Daily rainfall metrix.xlsx", "Daily rainfall metrics.xlsx", "Daily_rainfall_metrix.xlsx"]
    )
    if rain_path is None:
        raise FileNotFoundError("Could not find a rainfall metrics Excel file. Provide --rainfall-file.")
    print(f"[INFO] Rainfall metrics: {rain_path}")

    # BW files
    bw_files: List[Path] = []
    for fn in ["BW chem all data 2012 2021.xlsx", "BW chem data 2022.xlsx", "BW chem data 2023.xlsx", "BW chem data 2024.xlsx"]:
        p = base / fn
        if p.exists():
            bw_files.append(p)
    if not bw_files:
        bw_files = list(base.glob("BW chem*.xlsx")) + list(base.glob("BW_chem*.xlsx"))
    if not bw_files:
        raise FileNotFoundError("No BW chemistry Excel files found in base-dir. Expected 'BW chem ...xlsx' files.")
    print("[INFO] BW chemistry files:")
    for p in bw_files:
        print("   ", p.name)

    # Step4 packages
    packages_path: Optional[Path] = None
    if args.packages_file:
        packages_path = Path(args.packages_file)
    else:
        direct = base / "historical_packages.csv"
        if direct.exists():
            packages_path = direct
        else:
            latest = detect_latest_matching(base, "outputs_step4_packages_*")
            if latest is not None and (latest / "historical_packages.csv").exists():
                packages_path = latest / "historical_packages.csv"

    if packages_path is not None and packages_path.exists():
        print(f"[INFO] Step4 packages: {packages_path}")
    else:
        print("[WARN] No Step4 historical_packages.csv found. Models F/G/H/I/J will be skipped (need wind/UV/SSRD/T/soil/API).")
        packages_path = None

    # --- Output folder (always new)
    out_root = Path(args.out_dir) if args.out_dir else base
    safe_mkdir(out_root)
    out_dir = out_root / f"outputs_step5_models_{args.year_min}-{args.year_max}_May-Sep_{_ts()}"
    safe_mkdir(out_dir)
    print(f"[INFO] Output folder: {out_dir}")
    print(f"[INFO] Analysis period: {args.year_min}-{args.year_max}, months: {months}")

    # --- Load data
    rain = load_rainfall_metrics(rain_path, region=args.region)
    rain = rain[rain["month"].isin(months)].copy()
    rain = rain[(rain["year"] >= args.year_min) & (rain["year"] <= args.year_max)].copy()

    bw = load_bw_chemistry(bw_files)
    ydf = make_exceedance_dataset(bw, threshold=args.exceed_threshold, months=months, target_level=args.target_level)
    ydf = ydf[(ydf["year"] >= args.year_min) & (ydf["year"] <= args.year_max)].copy()

    # Merge drivers onto Y
    merged = ydf.merge(rain, on="date", how="left", suffixes=("", "_rain"))
    if packages_path is not None:
        packages = load_historical_packages(packages_path)
        merged = merged.merge(packages, on="date", how="left", suffixes=("", "_pkg"))

    # intensity preference
    if "intensity_q95_mmph" not in merged.columns and "q95_mmph" in merged.columns:
        merged["intensity_q95_mmph"] = merged["q95_mmph"]
    if "intensity_q95_mmph" not in merged.columns:
        merged["intensity_q95_mmph"] = merged.get("intensity_mean_mmph", np.nan)

    # depth
    if "depth_mm" not in merged.columns:
        if "daily_depth_mm" in merged.columns:
            merged["depth_mm"] = merged["daily_depth_mm"]
        elif "rain_mm" in merged.columns:
            merged["depth_mm"] = merged["rain_mm"]

    # family
    if "family" not in merged.columns:
        merged["family"] = merged["regime"].map(REGIME_TO_GROUP_4).astype("category")

    merged["month"] = merged["date"].dt.month
    merged = merged[merged["month"].isin(months)].copy()
    merged = merged[(merged["year"] >= args.year_min) & (merged["year"] <= args.year_max)].copy()
    merged = merged[np.isfinite(merged["y_exceed"])].copy()

    # --- Define model specs A–J (A–G unchanged; add H/I/J)
    specs: List[ModelSpec] = [
        ModelSpec(name="A_intensity_only", features_num=["intensity_q95_mmph"], features_cat=[]),
        ModelSpec(name="B_depth_only", features_num=["depth_mm"], features_cat=[]),
        ModelSpec(name="C_intensity_plus_depth", features_num=["intensity_q95_mmph", "depth_mm"], features_cat=[]),
        ModelSpec(name="D_WP", features_num=[], features_cat=["family"]),
        ModelSpec(name="E_WP_plus_intensity_plus_depth", features_num=["intensity_q95_mmph", "depth_mm"], features_cat=["family"]),
    ]

    # Extra drivers needed for F/G/H/I/J
    extra_cols = ["wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C", "swvl1_roll", "p7_mm", "api"]
    have_extra = packages_path is not None and all(c in merged.columns for c in extra_cols)

    soil_api_cols = ["swvl1_roll", "p7_mm", "api"]
    wind_rad_temp_cols = ["wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C"]

    if have_extra:
        specs += [
            ModelSpec(
                name="F_WP_plus_wind_UV_SSRD_t2m_soil_API",
                features_num=extra_cols,
                features_cat=["family"],
            ),
            ModelSpec(
                name="G_full_WP_plus_depth_intensity_plus_wind_UV_SSRD_t2m_soil_API",
                features_num=["depth_mm", "intensity_q95_mmph"] + extra_cols,
                features_cat=["family"],
            ),
            ModelSpec(
                name="H_soil_API_only",
                features_num=soil_api_cols,
                features_cat=[],
            ),
            ModelSpec(
                name="I_wind_UV_SSRD_t2m_only",
                features_num=wind_rad_temp_cols,
                features_cat=[],
            ),
            ModelSpec(
                name="J_wind_UV_SSRD_t2m_plus_soil_API",
                features_num=wind_rad_temp_cols + soil_api_cols,
                features_cat=[],
            ),
        ]
    else:
        print("[WARN] Skipping models F/G/H/I/J (missing some of wind/UV/SSRD/t2m/soil/API columns).")

    # Ensure feature columns exist
    for s in specs:
        for c in s.features_num + s.features_cat:
            if c not in merged.columns:
                raise ValueError(
                    f"Missing required feature column '{c}' for model {s.name}. "
                    f"Available: {list(merged.columns)}"
                )

    # --- Fit + evaluate (LOYO)
    y = merged["y_exceed"].astype(int).to_numpy()
    perf_rows = []
    pred_oof_raw: Dict[str, np.ndarray] = {}
    pred_oof_cal: Dict[str, np.ndarray] = {}

    for s in specs:
        feat_cols = s.features_num + s.features_cat
        pipe = build_pipeline(s.features_num, s.features_cat, seed=args.seed)

        p_oof_raw_s, p_oof_cal_s = lo_year_out_cv_predict_calibrated(
            merged, "y_exceed", "year", pipe, feat_cols, method=args.calibration_method
        )
        p_oof = p_oof_raw_s.to_numpy()
        p_oof_cal = p_oof_cal_s.to_numpy()

        pred_oof_raw[s.name] = p_oof
        pred_oof_cal[s.name] = p_oof_cal

        mask = np.isfinite(p_oof)
        if mask.sum() < 20:
            continue

        aucv = roc_auc_score(y[mask], p_oof[mask])
        brier_raw = brier_score_loss(y[mask], p_oof[mask])
        brier_cal = brier_score_loss(y[mask], p_oof_cal[mask])

        perf_rows.append({
            "model": s.name,
            "n": int(mask.sum()),
            "auc": float(aucv),
            "brier_raw": float(brier_raw),
            f"brier_cal_{args.calibration_method}": float(brier_cal),
        })

    perf = pd.DataFrame(perf_rows).sort_values("auc", ascending=False)
    perf.to_csv(out_dir / "model_performance_loyo.csv", index=False)

    # Pretty labels for figures
    LABEL_MAP = {
        "A_intensity_only": "A: intensity",
        "B_depth_only": "B: depth",
        "C_intensity_plus_depth": "C: intensity + depth",
        "D_WP": "D: WP",
        "E_WP_plus_intensity_plus_depth": "E: WP + intensity + depth",
        "F_WP_plus_wind_UV_SSRD_t2m_soil_API": "F: WP + wind + UV + SSRD + t2m + soil/API",
        "G_full_WP_plus_depth_intensity_plus_wind_UV_SSRD_t2m_soil_API": "G: WP + depth + intensity + wind + UV + SSRD + t2m + soil/API",
        "H_soil_API_only": "H: soil / API",
        "I_wind_UV_SSRD_t2m_only": "I: wind + UV + SSRD + t2m",
        "J_wind_UV_SSRD_t2m_plus_soil_API": "J: wind + UV + SSRD + t2m + soil/API",
    }

    plot_roc_curves(out_dir / "roc_curves_swew.png", y, pred_oof_raw, label_map=LABEL_MAP, legend_fontsize=8)
    plot_calibration_deciles(
        out_dir / "calibration_deciles_swew_raw.png",
        y,
        pred_oof_raw,
        title="Calibration (deciles) — SWEW (IE ≥ 63) — raw model output",
        label_map=LABEL_MAP,
        legend_fontsize=8,
    )
    plot_calibration_deciles(
        out_dir / f"calibration_deciles_swew_calibrated_{args.calibration_method}.png",
        y,
        pred_oof_cal,
        title=f"Calibration (deciles) — SWEW (IE ≥ 63) — calibrated ({args.calibration_method}; LOYO-safe)",
        label_map=LABEL_MAP,
        legend_fontsize=8,
    )
    plot_depth_exceed_by_family(out_dir / "depth_exceed_by_family.png", merged, "y_exceed", thr=args.exceed_threshold)

    merged.to_csv(out_dir / "modelling_dataset_swew.csv", index=False)

    # --- Optional: burden metrics + probability mastersheet on simulated drivers
    if args.sim_drivers_file:
        sim_path = Path(args.sim_drivers_file)
        if not sim_path.exists():
            raise FileNotFoundError(sim_path)

        sim = pd.read_csv(sim_path)
        sim = normalise_columns(sim)
        sim["date"] = pd.to_datetime(sim["date"])
        sim = sim[sim["date"].dt.month.isin(months)].copy()
        if sim["date"].dt.year.between(args.year_min, args.year_max).any():
            sim = sim[sim["date"].dt.year.between(args.year_min, args.year_max)].copy()

        if "family" not in sim.columns and "regime" in sim.columns:
            sim["family"] = sim["regime"].map(REGIME_TO_GROUP_4).astype("category")
        if "depth_mm" not in sim.columns and "rain_mm" in sim.columns:
            sim["depth_mm"] = sim["rain_mm"]
        if "intensity_q95_mmph" not in sim.columns and "q95_mmph" in sim.columns:
            sim["intensity_q95_mmph"] = sim["q95_mmph"]

        burden_rows = []

        for s in specs:
            pipe = build_pipeline(s.features_num, s.features_cat, seed=args.seed)
            feat_cols = s.features_num + s.features_cat

            # fit model on all observed
            pipe.fit(merged[feat_cols], merged["y_exceed"].astype(int))

            # raw prob on sim
            p_sim_raw = pipe.predict_proba(sim[feat_cols])[:, 1]
            sim[f"p_{s.name}"] = p_sim_raw

            # calibrate using observed (post-hoc)
            try:
                score_obs = pipe.decision_function(merged[feat_cols])
                score_sim = pipe.decision_function(sim[feat_cols])
            except Exception:
                eps = 1e-6
                p_obs = np.clip(pipe.predict_proba(merged[feat_cols])[:, 1], eps, 1 - eps)
                p_sim_clip = np.clip(p_sim_raw, eps, 1 - eps)
                score_obs = np.log(p_obs / (1 - p_obs))
                score_sim = np.log(p_sim_clip / (1 - p_sim_clip))

            if args.calibration_method.lower() == "sigmoid":
                cal = LogisticRegression(max_iter=2000, solver="lbfgs")
                cal.fit(np.asarray(score_obs).reshape(-1, 1), merged["y_exceed"].astype(int).to_numpy())
                p_sim_cal = cal.predict_proba(np.asarray(score_sim).reshape(-1, 1))[:, 1]
            else:
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(np.asarray(score_obs), merged["y_exceed"].astype(int).to_numpy())
                p_sim_cal = cal.transform(np.asarray(score_sim))

            sim[f"pcal_{s.name}"] = p_sim_cal

            # aggregate burden by storyline + season keys if present
            group_cols = []
            for c in ["storyline", "sim", "sim_id", "season_id", "year"]:
                if c in sim.columns:
                    group_cols.append(c)

            if not group_cols:
                m = compute_burden_metrics(sim[f"pcal_{s.name}"].to_numpy(), threshold=args.high_risk_threshold)
                m.update({"model": s.name})
                burden_rows.append(m)
            else:
                for key, sub in sim.groupby(group_cols):
                    sub = sub.sort_values("date")
                    pseq = sub[f"p_{s.name}"].to_numpy()
                    m = compute_burden_metrics(pseq, threshold=args.high_risk_threshold)
                    if isinstance(key, tuple):
                        for kcol, kval in zip(group_cols, key):
                            m[kcol] = kval
                    else:
                        m[group_cols[0]] = key
                    m["model"] = s.name
                    burden_rows.append(m)

        burden = pd.DataFrame(burden_rows)
        burden.to_csv(out_dir / "burden_metrics_simulated.csv", index=False)
        print(f"[INFO] Wrote burden metrics: {out_dir / 'burden_metrics_simulated.csv'}")

        # --- NEW: write probability mastersheet (A–J) if requested
        if args.save_sim_mastersheet:
            prob_dir = out_root / f"probabilities for models A - J mastersheet_{_ts()}"
            safe_mkdir(prob_dir)

            # master daily sheet: drivers + probabilities
            master_path = prob_dir / "sim_with_predictions.csv.gz"
            sim.to_csv(master_path, index=False, compression="gzip")

            # probs-only: keys + probs
            key_cols = [c for c in ["storyline", "sim", "sim_id", "season_id", "year", "date", "day_of_season"] if c in sim.columns]
            prob_cols = [c for c in sim.columns if c.startswith("p_") or c.startswith("pcal_")]
            probs_only = sim[key_cols + prob_cols].copy()
            probs_only_path = prob_dir / "probs_only.csv.gz"
            probs_only.to_csv(probs_only_path, index=False, compression="gzip")

            # small README
            readme = prob_dir / "README.txt"
            readme.write_text(
                "This folder was produced by Step5 (A–J).\n"
                "- sim_with_predictions.csv.gz: simulated daily drivers + p_* (raw) + pcal_* (calibrated) probabilities.\n"
                "- probs_only.csv.gz: keys + probability columns only (useful for Step5C merge).\n"
                f"Calibration method: {args.calibration_method}\n"
            )

            print(f"[INFO] Wrote probability mastersheet folder:\n  {prob_dir}")
            print(f"[INFO] Master file:\n  {master_path}")
            print(f"[INFO] Probs-only file:\n  {probs_only_path}")

    print("[DONE] Step 5 outputs written to:")
    print("  ", out_dir)


if __name__ == "__main__":
    main()
