# trend_histograms_seasons_maysep_by_regime_NOQC_with_excel.py
# NO QC: use ALL raw daily histograms.
# Scope: per REGION × REGIME for
#   (1) Meteorological seasons (DJF, MAM, JJA, SON)
#   (2) May–September (inclusive)
# Outputs:
#   - PNGs in NEW folders (won't overwrite your previous ones)
#   - Per-figure sidecar CSVs with the exact percentiles used
#   - One Excel workbook with percentiles (long & wide), daily metrics (log-mean, log-space), and bins metadata.

import os, re, glob, math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PATHS ----------------
ROOT             = r"C:\Users\earth\OneDrive - University of Reading\Regime data QC - log mean - no QC - raw data"
HIST_GLOB        = os.path.join(ROOT, "**", "rainfall_linear_all_regions_*.csv")  # recursive
CLASS_CSV        = os.path.join(ROOT, "classifications_30regimes.csv")

# NEW folders (so you don't overwrite earlier outputs)
OUT_DIR_SEAS_R   = os.path.join(ROOT, "histograms meteo seasons by regime - NOQC")
OUT_DIR_MAYSEP_R = os.path.join(ROOT, "histograms May-Sep by regime - NOQC")
EXCEL_DIR        = os.path.join(ROOT, "histogram stats by regime - NOQC")
EXCEL_PATH       = os.path.join(EXCEL_DIR, "seasons_maysep_by_regime_NOQC_stats.xlsx")
os.makedirs(OUT_DIR_SEAS_R, exist_ok=True)
os.makedirs(OUT_DIR_MAYSEP_R, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

# ---------------- SETTINGS ----------------
REGIMES_TO_INCLUDE = set(range(1, 31))   # 1..30
MIN_DAYS_PER_GROUP = 1
USE_LOG_Y          = True
Y_FLOOR            = 1.0
TITLE_PREFIX       = "Intensity Distribution"

# Physical metrics (identical basis to QC workflow)
TOP_BIN_FACTOR = 2.0     # pseudo upper edge for the open-ended top bin (e.g., 256→512)
WET_EPS        = 0.0     # edges <= WET_EPS = "dry" for wet-only metrics

# Canonical bins (mm/h lower edges) – strict, ordered columns
CANON_BINS = [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]

# ---------------- HELPERS ----------------
DATE_IN_NAME = re.compile(r"(\d{8})")

def parse_date_from_filename(fname: str):
    m = DATE_IN_NAME.search(fname)
    if not m:
        return None
    dt = pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
    return None if pd.isna(dt) else dt.normalize().date()

def parse_date_series_smart(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    if s.str.match(r"\d{2}/\d{2}/\d{4}$").mean() > 0.8:
        return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    if s.str.match(r"\d{4}-\d{2}-\d{2}$").mean() > 0.8:
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if s.str.match(r"\d{8}$").mean() > 0.8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, dayfirst=True, infer_datetime_format=True, errors="coerce")

def clean_region_name(region):
    s = re.sub(r"[\\/]+", "_", str(region))
    s = re.sub(r"\s+", "_", s.strip())
    return s

def month_to_season(m: int) -> str:
    if m in (12, 1, 2):   return "DJF"
    if m in (3, 4, 5):    return "MAM"
    if m in (6, 7, 8):    return "JJA"
    if m in (9, 10, 11):  return "SON"
    return "NA"

def load_regime_map(class_csv):
    reg = pd.read_csv(class_csv)
    cols = {c.lower(): c for c in reg.columns}
    date_col = next((cols[c] for c in cols if "date" in c), None)
    dt = None
    if date_col:
        dt = parse_date_series_smart(reg[date_col]).dt.normalize()
    if dt is None or dt.isna().mean() > 0.2:
        y = cols.get("year"); m = cols.get("month"); d = cols.get("day")
        if y and m:
            day_series = pd.to_numeric(reg[d], errors="coerce") if d else 1
            dt = pd.to_datetime(
                dict(year=pd.to_numeric(reg[y], errors="coerce"),
                     month=pd.to_numeric(reg[m], errors="coerce"),
                     day=day_series),
                errors="coerce"
            ).dt.normalize()
    if dt is None or dt.isna().all():
        raise ValueError("Could not detect a usable date in classifications_30regimes.csv")
    reg["_date_"] = dt

    r = cols.get("regime")
    if r is None:
        best, score = None, -1
        for c in reg.columns:
            vals = pd.to_numeric(reg[c], errors="coerce")
            ok = vals.dropna()
            if len(ok) < 10: continue
            frac_int = ((ok % 1) == 0).mean()
            in_range = ok.between(1, 30).mean()
            s = 0.5*frac_int + 0.5*in_range
            if s > score: score, best = s, c
        r = best
    reg["_regime_"] = pd.to_numeric(reg[r], errors="coerce").astype("Int64")
    reg = reg.dropna(subset=["_date_", "_regime_"]).drop_duplicates(subset=["_date_"], keep="first")
    return dict(zip(reg["_date_"].dt.date.values, reg["_regime_"].astype(int).values))

def _safe_float_eq(colname: str, target: float) -> bool:
    try:
        return float(colname) == float(target)
    except Exception:
        return False

# ---- grouped-data utilities (same science as QC) ----
def logarithmic_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0: return 0.0
    if a == b: return float(a)
    return float((b - a) / (math.log(b) - math.log(a)))

def bin_edges_and_reps(canon_bins: List[float], top_factor: float=2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.asarray(canon_bins, dtype=float)
    uppers = np.r_[edges[1:], edges[-1]*top_factor]
    reps   = np.array([0.0 if a <= 0 else logarithmic_mean(a, b) for a,b in zip(edges, uppers)], dtype=float)
    return edges, uppers, reps

def grouped_quantile(edges: np.ndarray, uppers: np.ndarray, weights: np.ndarray, q: float) -> float:
    w = np.asarray(weights, dtype=float)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0: return 0.0
    cw = np.cumsum(w); target = q * tot
    idx = int(np.searchsorted(cw, target, side="left"))
    idx = min(max(idx, 0), len(w)-1)
    w_i = w[idx]
    if w_i <= 0:
        nz = np.flatnonzero(w)
        if len(nz) == 0: return 0.0
        idx = int(nz[min(np.searchsorted(nz, idx), len(nz)-1)])
        w_i = w[idx]
    lower = float(edges[idx]); upper = float(uppers[idx])
    prev_cum = cw[idx] - w_i
    frac = float(np.clip((target - prev_cum) / w_i if w_i > 0 else 0.0, 0.0, 1.0))
    lo = max(lower, np.finfo(float).tiny)
    hi = max(upper, lo*(1.0+1e-12))
    return float(np.exp(np.log(lo) + frac*(np.log(hi) - np.log(lo))))

# ---------------- DATA GATHER ----------------
def strict_row_to_canonical(row: pd.Series, float_to_col: dict) -> np.ndarray:
    out = np.zeros(len(CANON_BINS), dtype=float)
    for j, b in enumerate(CANON_BINS):
        if str(b) in row.index:
            val = row[str(b)]
        elif b in float_to_col:
            val = row[float_to_col[b]]
        else:
            val = 0.0
        try:
            out[j] = float(val)
        except Exception:
            out[j] = 0.0
    return out

def collect_grouped_data(glob_pattern, regime_map, regimes_wanted):
    """
    NO QC. Returns:
      data_season_reg[(region, regime, season)] -> [counts_dict_per_day, ...]
      data_maysep_reg[(region, regime)]         -> [counts_dict_per_day, ...]
      days_used_*  : Counters
      daily_rows   : per-day physical metrics using log-mean reps (for Excel)
    """
    files = sorted(glob.glob(glob_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No histogram files matched: {glob_pattern}")

    edges, uppers, reps = bin_edges_and_reps(CANON_BINS, TOP_BIN_FACTOR)

    data_season_reg   = defaultdict(list)
    data_maysep_reg   = defaultdict(list)
    days_used_season  = Counter()
    days_used_maysep  = Counter()
    daily_rows        = []

    for fp in files:
        date = parse_date_from_filename(os.path.basename(fp))
        if date is None:
            continue
        regime = regime_map.get(date)
        if regime is None or regime not in regimes_wanted:
            continue

        month  = int(pd.Timestamp(date).month)
        season = month_to_season(month)

        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if "region" not in df.columns:
            continue
        df["region"] = df["region"].astype(str).str.strip()

        # float-equal mapping for "0.5" vs "0.500000"
        bin_cols = [c for c in df.columns if c != "region"]
        float_to_col = {}
        for c in bin_cols:
            try:
                float_to_col[float(c)] = c
            except Exception:
                pass

        for _, row in df.iterrows():
            region = row["region"]
            counts = strict_row_to_canonical(row, float_to_col)

            # store per-day dict(view) for percentile graphs
            data_season_reg[(region, regime, season)].append(dict(zip(CANON_BINS, counts)))
            days_used_season[(region, regime, season)] += 1
            if 5 <= month <= 9:
                data_maysep_reg[(region, regime)].append(dict(zip(CANON_BINS, counts)))
                days_used_maysep[(region, regime)] += 1

            # ---- daily physical metrics (same science as QC) ----
            S = float(np.nansum(counts))
            zero_mask = np.array(CANON_BINS, dtype=float) <= WET_EPS
            wet_fraction = float(1.0 - (np.nansum(counts[zero_mask]) / S)) if S > 0 else 0.0
            if S > 0:
                mean_rate = float(np.dot(reps, counts) / S)
                daily_depth_mm = mean_rate * 24.0
                wet_mask = np.array(CANON_BINS, dtype=float) > WET_EPS
                if wet_mask.any() and np.nansum(counts[wet_mask]) > 0:
                    q50 = grouped_quantile(edges[wet_mask], uppers[wet_mask], counts[wet_mask], 0.50)
                    q90 = grouped_quantile(edges[wet_mask], uppers[wet_mask], counts[wet_mask], 0.90)
                    q95 = grouped_quantile(edges[wet_mask], uppers[wet_mask], counts[wet_mask], 0.95)
                else:
                    q50 = q90 = q95 = 0.0
            else:
                mean_rate = daily_depth_mm = q50 = q90 = q95 = 0.0

            daily_rows.append({
                "date": pd.to_datetime(date),
                "month": month,
                "season": season,
                "region": region,
                "regime": int(regime),
                "total_samples": S,
                "wet_fraction": wet_fraction,
                "q50_mmph": q50,
                "q90_mmph": q90,
                "q95_mmph": q95,
                "mean_rate_mmph": mean_rate,
                "daily_depth_mm": daily_depth_mm
            })

    return data_season_reg, data_maysep_reg, days_used_season, days_used_maysep, daily_rows

def percentiles_matrix(dict_list, bins_sorted, ps=(10,25,50,75,90,95,99,100)):
    n_days = len(dict_list)
    mat = np.zeros((n_days, len(bins_sorted)), dtype=float)
    for i, d in enumerate(dict_list):
        for j, b in enumerate(bins_sorted):
            v = d.get(b, 0.0)
            mat[i, j] = 0.0 if (v is None or not np.isfinite(v)) else float(v)
    out = {}
    for p in ps:
        out[f"P{p}"] = np.percentile(mat, p, axis=0)
    return out

# ---------------- PLOT + STATS SAVE ----------------
def plot_trend(title_suffix, bins_sorted, percs, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    x = np.arange(len(bins_sorted))
    labels = [str(int(v)) if float(v).is_integer() else str(v) for v in bins_sorted]
    p10,p25,p50,p75,p90,p95,p99,p100 = (percs["P10"], percs["P25"], percs["P50"], percs["P75"],
                                        percs["P90"], percs["P95"], percs["P99"], percs["P100"])
    if USE_LOG_Y:
        floor = lambda a: np.where(a>0, a, Y_FLOOR)
        p10,p25,p50,p75,p90,p95,p99,p100 = map(floor, (p10,p25,p50,p75,p90,p95,p99,p100))
    plt.figure(figsize=(12,6))
    plt.fill_between(x, p25, p75, alpha=0.35, label="p25–p75", color="C0")
    plt.fill_between(x, p10, p90, alpha=0.25, label="p10–p90", color="C2")
    plt.plot(x, p50,  color="C0",  lw=2.2, label="Median (p50)")
    plt.plot(x, p95,  color="red",    lw=1.8, ls="--", label="p95")
    plt.plot(x, p99,  color="purple", lw=1.6, ls=":",  label="p99")
    plt.plot(x, p100, color="black",  lw=1.4, ls=":",  label="p100")
    if USE_LOG_Y: plt.yscale("log")
    plt.xticks(x, labels)
    plt.xlabel("Bins (mm/h lower edge)")
    plt.ylabel("Count (percentiles across days)")
    plt.title(f"{TITLE_PREFIX} – {title_suffix}")
    plt.grid(True, which="both", alpha=0.25, linestyle=":")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def save_sidecar_csv(out_png, bins_sorted, percs, meta: Dict[str, object]):
    out_csv = out_png.replace(".png", ".csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame({
        "bin_lower_mmph": bins_sorted,
        "P10":  percs["P10"], "P25": percs["P25"], "P50": percs["P50"], "P75": percs["P75"],
        "P90":  percs["P90"], "P95": percs["P95"], "P99": percs["P99"], "P100": percs["P100"],
    })
    # prepend meta columns
    cols = list(df.columns)
    for k in ["season","region","regime","n_days"]:
        if k in meta:
            df.insert(0, k, meta[k])
    df.to_csv(out_csv, index=False)

# ---------------- MAIN ----------------
def main():
    # bins metadata for Excel
    edges, uppers, reps = bin_edges_and_reps(CANON_BINS, TOP_BIN_FACTOR)
    bins_meta = pd.DataFrame({
        "bin_lower_mmph": edges,
        "bin_upper_mmph": uppers,
        "rep_logmean_mmph": reps
    })

    # load regimes & group data (NO QC)
    regime_map = load_regime_map(CLASS_CSV)
    (data_season_reg, data_maysep_reg,
     days_used_season, days_used_maysep,
     daily_rows) = collect_grouped_data(HIST_GLOB, regime_map, REGIMES_TO_INCLUDE)

    # master tables for Excel
    season_long = []
    maysep_long = []

    # ---- Meteorological seasons (by region × regime) ----
    total_season = 0
    season_order = {"DJF":0, "MAM":1, "JJA":2, "SON":3}
    for (region, regime, season), dict_list in sorted(
            data_season_reg.items(),
            key=lambda kv: (clean_region_name(kv[0][0]), kv[0][1], season_order.get(kv[0][2], 99))):
        n_days = days_used_season[(region, regime, season)]
        if n_days < MIN_DAYS_PER_GROUP:
            continue
        percs = percentiles_matrix(dict_list, CANON_BINS)
        out_png = os.path.join(
            OUT_DIR_SEAS_R, clean_region_name(region),
            f"{clean_region_name(region)}_regime_{int(regime):02d}_{season}.png"
        )
        title_suffix = f"{region} (Regime {int(regime):02d}) – {season}"
        plot_trend(title_suffix, CANON_BINS, percs, out_png)
        save_sidecar_csv(out_png, CANON_BINS, percs, {
            "season": season, "region": region, "regime": int(regime), "n_days": int(n_days)
        })
        for j, b in enumerate(CANON_BINS):
            for stat, arr in percs.items():
                season_long.append({
                    "season": season, "region": region, "regime": int(regime),
                    "n_days": int(n_days), "bin_lower_mmph": b,
                    "stat": stat, "value": float(arr[j])
                })
        total_season += 1

    # ---- May–September (by region × regime) ----
    total_maysep = 0
    for (region, regime), dict_list in sorted(
            data_maysep_reg.items(),
            key=lambda kv: (clean_region_name(kv[0][0]), kv[0][1])):
        n_days = days_used_maysep[(region, regime)]
        if n_days < MIN_DAYS_PER_GROUP:
            continue
        percs = percentiles_matrix(dict_list, CANON_BINS)
        out_png = os.path.join(
            OUT_DIR_MAYSEP_R, clean_region_name(region),
            f"{clean_region_name(region)}_regime_{int(regime):02d}_May-Sep.png"
        )
        title_suffix = f"{region} (Regime {int(regime):02d}) – May–Sep"
        plot_trend(title_suffix, CANON_BINS, percs, out_png)
        save_sidecar_csv(out_png, CANON_BINS, percs, {
            "region": region, "regime": int(regime), "n_days": int(n_days)
        })
        for j, b in enumerate(CANON_BINS):
            for stat, arr in percs.items():
                maysep_long.append({
                    "region": region, "regime": int(regime),
                    "n_days": int(n_days), "bin_lower_mmph": b,
                    "stat": stat, "value": float(arr[j])
                })
        total_maysep += 1

    # ---- Build Excel sheets ----
    season_long_df = pd.DataFrame(season_long)
    maysep_long_df = pd.DataFrame(maysep_long)
    def wide(df):
        if df.empty: return pd.DataFrame()
        return (df.pivot_table(index=[c for c in df.columns if c not in ("stat","value")],
                               columns="stat", values="value", aggfunc="first")
                  .reset_index())

    season_wide_df = wide(season_long_df)
    maysep_wide_df = wide(maysep_long_df)

    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df.sort_values(["region","regime","date"], inplace=True)

    with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as xl:
        bins_meta.to_excel(xl, sheet_name="bins_metadata", index=False)
        season_long_df.to_excel(xl, sheet_name="season_percentiles_long", index=False)
        season_wide_df.to_excel(xl, sheet_name="season_percentiles_wide", index=False)
        maysep_long_df.to_excel(xl, sheet_name="MaySep_percentiles_long", index=False)
        maysep_wide_df.to_excel(xl, sheet_name="MaySep_percentiles_wide", index=False)
        daily_df.to_excel(xl, sheet_name="daily_metrics", index=False)

    print(f"[done] Season (by regime) plots: {total_season}  → {OUT_DIR_SEAS_R}")
    print(f"[done] May–Sep (by regime) plots: {total_maysep} → {OUT_DIR_MAYSEP_R}")
    print(f"[done] Excel workbook → {EXCEL_PATH}")

if __name__ == "__main__":
    main()
