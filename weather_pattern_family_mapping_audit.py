"""
Reviewer-auditable storyline grouping for SWEW (May–Sep, NOQC).

INPUT:
  SWEW_MaySep_by_regime_NOQC_stats.xlsx  (must contain sheet: 'daily_metrics')

OUTPUTS (new folder next to input workbook):
  - daily_metrics_filtered.csv
  - regime_metrics_by_regime.xlsx / .csv
  - storyline_thresholds.csv / .json
  - storyline_assignments.xlsx / .csv
  - rule_flags_by_regime.csv
  - AUDIT_storylines.xlsx   (all intermediate tables in one place)
  - README_storylines.txt

METHOD (why these steps):
  1) Filter to SWEW and May–Sep to match study scope.
  2) Summarise each regime using robust percentiles (not means), because rainfall is skewed.
     - depth_p90_mm: wet-day amount signal (frontal/large-scale contribution)
     - tail_q95_p90_mmph: upper tail strength (embedded convection / bursts)
     - wf_p50: rainfall footprint extent (widespread vs localised)
  3) Define thresholds objectively via quantiles across the 30 regime summaries.
     This is scale-free and reproducible.
  4) Assign regimes to one of six storylines via deterministic rules, and log which rule fired.

References to justify percentile/quantile use in methods:
- ETCCDI / HadEX precipitation extremes indices use percentile thresholds (R95p/R99p).
- Wilks (Statistical Methods in the Atmospheric Sciences) for quantiles & nonparametric robustness.
- Ferro (2004) for quantiles highlighting tail differences in meteorological distributions.
- Hannachi et al. (2017) for circulation regimes framework.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# PATHS (EDIT)
# -----------------------------
REGIME_STATS_XLSX = Path(
    r"C:\Users\earth\OneDrive - University of Reading\Python codes objective 2\Beyond_rainfall_triggers\Rainfall fingerprints\SWEW histogram stats by regime - NOQC\SWEW_MaySep_by_regime_NOQC_stats.xlsx"
)

OUT_FOLDER_NAME = "storylines_swew_maysep_noqc_audit 1"


# -----------------------------
# SCOPE
# -----------------------------
REGION = "SWEW"
MONTHS = {5, 6, 7, 8, 9}  # May–Sep


# -----------------------------
# THRESHOLD QUANTILES (across regimes)
# -----------------------------
Q_WF_HIGH = 0.80       # top quintile wet-fraction => widespread rainfall footprint
Q_WF_LOW  = 0.40       # lower extent group
Q_TAIL_HIGH = 0.80     # top quintile tail strength => convective-like extremes signal
Q_TAIL_LOW  = 0.40     # weak tails
Q_DEPTH_HIGH = 0.60    # upper 40% wet-day amounts (depth_p90)
Q_DEPTH_VHIGH = 0.80   # upper 20% wet-day amounts (depth_p90)
Q_DEPTH_MEDLOW = 0.40  # low typical background (depth_p50)



# -----------------------------
# FINAL 4-GROUP STORYLINES (EXPERT-REVIEWED)
# -----------------------------
# These are the final storyline labels used in the manuscript figures/tables.
# They are defined directly at the WP (1–30) level for reviewer clarity.
#
# NOTE: Following meteorological review, WP16 is treated as "Convective extremes"
# (low wet fraction but strong tail), and WP28 is treated as
# "Cyclonic Atlantic (frontal)" (cyclonic low with near-threshold wet footprint).
#
# If you want to reproduce the *rule-based* (threshold-only) grouping, use `group_rule6`
# and `group_rule4` columns in the outputs. The manuscript uses `group_final4`.

FINAL_GROUPS_4 = [
    "Cyclonic Atlantic (frontal)",
    "Showery maritime / unsettled",
    "Convective extremes",
    "Settled anticyclonic quiet",
]

REGIME_TO_GROUP_4_FINAL: dict[int, str] = {
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
    16: "Convective extremes",              # expert-reviewed
    17: "Convective extremes",
    18: "Convective extremes",
    19: "Cyclonic Atlantic (frontal)",      # expert-reviewed
    20: "Showery maritime / unsettled",
    21: "Cyclonic Atlantic (frontal)",
    22: "Cyclonic Atlantic (frontal)",
    23: "Settled anticyclonic quiet",
    24: "Cyclonic Atlantic (frontal)",
    25: "Settled anticyclonic quiet",
    26: "Showery maritime / unsettled",
    27: "Convective extremes",
    28: "Cyclonic Atlantic (frontal)",     # expert-reviewed
    29: "Cyclonic Atlantic (frontal)",
    30: "Cyclonic Atlantic (frontal)",
}

REGIME_TO_SYNOPTIC_LABEL: dict[int, str] = {
    1: "Unbiased northwesterly",
    2: "Cyclonic southwesterly (returning polar maritime)",
    3: "Anticyclonic southwesterly (ridge over N France)",
    4: "Unbiased westerly",
    5: "Unbiased southerly (high over Scandinavia)",
    6: "Anticyclonic Azores High extension towards UK",
    7: "Cyclonic southwesterly (low WNW of Ireland)",
    8: "Cyclonic westerly (low near Shetland)",
    9: "Anticyclonic N–N-easterly (high near Iceland)",
    10: "Anticyclonic W–SW (slight Azores ridge)",
    11: "Cyclonic (low centred over southern UK)",
    12: "Anticyclonic southerly (high over Poland)",
    13: "Anticyclonic northwesterly (high SW of Ireland)",
    14: "Cyclonic N–NW (low near southern Sweden)",
    15: "Unbiased southwesterly; very windy in N Britain",
    16: "Anticyclonic S–SE (high east of Denmark)",
    17: "Anticyclonic E–SE (high over Denmark)",
    18: "Anticyclonic southwesterly (high over northern France)",
    19: "Unbiased northerly (low east of Denmark)",
    20: "Cyclonic westerly (intense low near Iceland)",
    21: "Cyclonic southwesterly (deep low south of Iceland)",
    22: "Cyclonic southerly (low west of Ireland)",
    23: "Unbiased westerly; very windy in the north",
    24: "Cyclonic northerly (low in the North Sea)",
    25: "Anticyclonic northerly (high in the Irish Sea)",
    26: "Cyclonic northwesterly (low near Norway; very windy)",
    27: "Anticyclonic easterly (high in Norwegian Sea)",
    28: "Cyclonic southeasterly (low SW of the UK)",
    29: "Cyclonic S–SW (deep low west of Ireland)",
    30: "Cyclonic W–SW (deep low SE of Iceland)",
}

def group4_from_rule6(g6: str) -> str:
    """Collapse the rule-based 6-storyline scheme to 4 groups used in the manuscript."""
    g6 = str(g6)
    if g6 in {"Widespread cyclonic deluges", "Frontal Atlantic lows"}:
        return "Cyclonic Atlantic (frontal)"
    if g6 in {"Localised convective extremes", "Anticyclonic breakdown storms"}:
        return "Convective extremes"
    if g6 == "Showery maritime / unsettled":
        return "Showery maritime / unsettled"
    if g6 == "Settled anticyclonic quiet":
        return "Settled anticyclonic quiet"
    return g6

# -----------------------------
# HELPERS
# -----------------------------
def pct(series: pd.Series, p: float) -> float:
    """Robust percentile helper (ignores NaNs)."""
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, p)) if arr.size else float("nan")


def build_regime_metrics(dm: pd.DataFrame) -> pd.DataFrame:
    """
    Regime summary table: one row per pattern (regime).

    We use:
      depth_p90_mm         = P90(daily_depth_mm) across all May–Sep days in that regime
      tail_q95_p90_mmph    = P90(q95_mmph) across days in that regime
      wf_p50               = median(wet_fraction) across days in that regime

    Why P90 and median?
      - daily rainfall is highly skewed; percentiles are robust and tail-relevant.
    """
    required = {"date","month","region","regime","wet_fraction","q95_mmph","daily_depth_mm"}
    missing = required - set(dm.columns)
    if missing:
        raise ValueError(f"daily_metrics missing columns: {sorted(missing)}")

    dm = dm.copy()
    dm["date"] = pd.to_datetime(dm["date"])
    dm["month"] = pd.to_numeric(dm["month"], errors="coerce").astype("Int64")
    dm["regime"] = pd.to_numeric(dm["regime"], errors="coerce").astype("Int64")

    # Filter to paper scope
    dm = dm[(dm["region"].astype(str).str.strip() == REGION) & (dm["month"].isin(MONTHS))]
    dm = dm.dropna(subset=["regime"])

    regm = (
        dm.groupby("regime", as_index=False)
          .agg(
              n_days=("date", "count"),
              depth_p50_mm=("daily_depth_mm", lambda x: pct(x, 50)),
              depth_p90_mm=("daily_depth_mm", lambda x: pct(x, 90)),
              tail_q95_p90_mmph=("q95_mmph", lambda x: pct(x, 90)),
              wf_p50=("wet_fraction", lambda x: pct(x, 50)),
          )
          .sort_values("regime")
          .reset_index(drop=True)
    )
    return dm, regm


def compute_thresholds(regm: pd.DataFrame) -> pd.DataFrame:
    """
    Thresholds are quantiles computed *across the 30 regime summaries*.

    This makes the scheme:
      - objective (no hand-tuning)
      - scale-free (works if magnitudes shift)
      - reproducible (same input -> same thresholds)
    """
    thr = [
        ("wf_high", "wf_p50", Q_WF_HIGH, float(regm["wf_p50"].quantile(Q_WF_HIGH))),
        ("wf_low",  "wf_p50", Q_WF_LOW,  float(regm["wf_p50"].quantile(Q_WF_LOW))),
        ("tail_high","tail_q95_p90_mmph", Q_TAIL_HIGH, float(regm["tail_q95_p90_mmph"].quantile(Q_TAIL_HIGH))),
        ("tail_low", "tail_q95_p90_mmph", Q_TAIL_LOW,  float(regm["tail_q95_p90_mmph"].quantile(Q_TAIL_LOW))),
        ("depth_high","depth_p90_mm", Q_DEPTH_HIGH, float(regm["depth_p90_mm"].quantile(Q_DEPTH_HIGH))),
        ("depth_vhigh","depth_p90_mm", Q_DEPTH_VHIGH, float(regm["depth_p90_mm"].quantile(Q_DEPTH_VHIGH))),
        ("depth_medlow","depth_p50_mm", Q_DEPTH_MEDLOW, float(regm["depth_p50_mm"].quantile(Q_DEPTH_MEDLOW))),
    ]
    return pd.DataFrame(thr, columns=["threshold_name","metric","quantile","value"])


def assign_storyline_with_flags(row: pd.Series, thr: dict) -> tuple[str, str, dict]:
    """
    Deterministic assignment with explicit flags, so reviewers can see exactly why.

    Returns:
      group, reason, flags_dict
    """
    wf = float(row["wf_p50"])
    d50 = float(row["depth_p50_mm"])
    d90 = float(row["depth_p90_mm"])
    tail = float(row["tail_q95_p90_mmph"])

    flags = {
        "wf_ge_high": wf >= thr["wf_high"],
        "wf_le_low": wf <= thr["wf_low"],
        "tail_ge_high": tail >= thr["tail_high"],
        "tail_le_low": tail <= thr["tail_low"],
        "depth90_ge_high": d90 >= thr["depth_high"],
        "depth90_ge_vhigh": d90 >= thr["depth_vhigh"],
        "depth50_le_medlow": d50 <= thr["depth_medlow"],
    }

    # 1) Widespread cyclonic deluges
    if flags["wf_ge_high"]:
        return ("Widespread cyclonic deluges",
                f"wf_p50={wf:.3f} ≥ wf_high({thr['wf_high']:.3f}); widespread footprint with high wet-day amounts (depth_p90={d90:.2f} mm).",
                flags)

    # 5) Settled anticyclonic quiet
    if flags["depth50_le_medlow"] and flags["wf_le_low"] and flags["tail_le_low"]:
        return ("Settled anticyclonic quiet",
                f"depth_p50={d50:.3f} ≤ depth_medlow({thr['depth_medlow']:.3f}), wf_p50={wf:.3f} ≤ wf_low({thr['wf_low']:.3f}), tail={tail:.2f} ≤ tail_low({thr['tail_low']:.2f}).",
                flags)

    # 4/6) Low-extent regimes with strong tails
    if flags["wf_le_low"] and flags["depth50_le_medlow"]:
        if flags["tail_ge_high"]:
            if flags["depth90_ge_high"]:
                return ("Anticyclonic breakdown storms",
                        f"Low extent but impactful: wf_p50={wf:.3f} ≤ wf_low({thr['wf_low']:.3f}), tail={tail:.2f} ≥ tail_high({thr['tail_high']:.2f}), depth_p90={d90:.2f} ≥ depth_high({thr['depth_high']:.2f}).",
                        flags)
            return ("Localised convective extremes",
                    f"Low extent + very strong tail: wf_p50={wf:.3f} ≤ wf_low({thr['wf_low']:.3f}) and tail={tail:.2f} ≥ tail_high({thr['tail_high']:.2f}).",
                    flags)
        if tail > thr["tail_low"]:
            return ("Anticyclonic breakdown storms",
                    f"Typically dry/low extent (depth_p50={d50:.3f}, wf_p50={wf:.3f}) but tail elevated (tail={tail:.2f} > tail_low({thr['tail_low']:.2f})).",
                    flags)

    # 3) Showery maritime/unsettled vs 2) Frontal Atlantic lows
    if flags["depth90_ge_vhigh"] and wf < 0.08:
        return ("Showery maritime / unsettled",
                f"Very wet days (depth_p90={d90:.2f} ≥ depth_vhigh({thr['depth_vhigh']:.2f})) but only moderate extent (wf_p50={wf:.3f}) → intermittent/heavy showers.",
                flags)

    if flags["depth90_ge_high"]:
        return ("Frontal Atlantic lows",
                f"High wet-day amounts (depth_p90={d90:.2f} ≥ depth_high({thr['depth_high']:.2f})) with moderate extent (wf_p50={wf:.3f}); tail not in top quintile (tail={tail:.2f}).",
                flags)

    return ("Showery maritime / unsettled",
            f"Moderate amounts/extent (depth_p90={d90:.2f}, wf_p50={wf:.3f}) with showery behaviour implied by tail={tail:.2f}.",
            flags)


def main():
    if not REGIME_STATS_XLSX.exists():
        raise FileNotFoundError(f"Not found: {REGIME_STATS_XLSX}")

    out_dir = REGIME_STATS_XLSX.parent / OUT_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load daily metrics
    dm_raw = pd.read_excel(REGIME_STATS_XLSX, sheet_name="daily_metrics")

    # --- Filter + regime summaries
    dm_filt, regm = build_regime_metrics(dm_raw)

    # --- Threshold table (across regimes)
    thr_df = compute_thresholds(regm)
    thr = {r["threshold_name"]: float(r["value"]) for _, r in thr_df.iterrows()}

    # --- Assign with flags
    groups, reasons, flags_rows = [], [], []
    for _, row in regm.iterrows():
        g, r, flags = assign_storyline_with_flags(row, thr)
        groups.append(g)
        reasons.append(r)
        flags_rows.append(flags)

    regm_out = regm.copy()
    regm_out["group"] = groups

    # Rule-based collapse (6 -> 4) for comparison/audit
    regm_out["group_rule6"] = regm_out["group"]
    regm_out["group_rule4"] = regm_out["group_rule6"].apply(group4_from_rule6)

    # Manuscript final grouping (expert-reviewed WP-level mapping)
    regm_out["group_final4"] = regm_out["regime"].astype(int).map(REGIME_TO_GROUP_4_FINAL)
    regm_out["synoptic_label"] = regm_out["regime"].astype(int).map(REGIME_TO_SYNOPTIC_LABEL)
    if regm_out["group_final4"].isna().any():
        missing = regm_out.loc[regm_out["group_final4"].isna(), "regime"].tolist()
        raise ValueError(f"Missing FINAL group mapping for regimes: {missing}")
    regm_out["reason"] = reasons

    flags_df = pd.DataFrame(flags_rows)
    flags_df.insert(0, "regime", regm_out["regime"].values)

    # --- Save outputs
    dm_filt.to_csv(out_dir / "daily_metrics_filtered.csv", index=False)

    regm_out.to_csv(out_dir / "storyline_assignments.csv", index=False)
    regm_out.to_excel(out_dir / "storyline_assignments.xlsx", index=False)

    regm[["regime","n_days","depth_p50_mm","depth_p90_mm","tail_q95_p90_mmph","wf_p50"]].to_csv(
        out_dir / "regime_metrics_by_regime.csv", index=False
    )
    regm[["regime","n_days","depth_p50_mm","depth_p90_mm","tail_q95_p90_mmph","wf_p50"]].to_excel(
        out_dir / "regime_metrics_by_regime.xlsx", index=False
    )

    thr_df.to_csv(out_dir / "storyline_thresholds.csv", index=False)
    (out_dir / "storyline_thresholds.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")

    flags_df.to_csv(out_dir / "rule_flags_by_regime.csv", index=False)

    # --- Single audit workbook
    audit_xlsx = out_dir / "AUDIT_storylines.xlsx"
    with pd.ExcelWriter(audit_xlsx, engine="openpyxl") as w:
        dm_filt.to_excel(w, sheet_name="daily_metrics_filtered", index=False)
        regm.to_excel(w, sheet_name="regime_metrics", index=False)
        thr_df.to_excel(w, sheet_name="thresholds", index=False)
        flags_df.to_excel(w, sheet_name="rule_flags", index=False)
        regm_out.to_excel(w, sheet_name="storyline_assignments", index=False)

    # --- README
    readme = f"""SWEW May–Sep Storylines (NOQC) – Audit Trail

INPUT
  {REGIME_STATS_XLSX}

FILTER
  region = {REGION}
  months = {sorted(MONTHS)}

PER-REGIME METRICS (from daily_metrics_filtered)
  depth_p50_mm         = median(daily_depth_mm)
  depth_p90_mm         = P90(daily_depth_mm)
  tail_q95_p90_mmph    = P90(q95_mmph)
  wf_p50               = median(wet_fraction)

THRESHOLDS (quantiles across the 30 regimes)
FINAL 4-GROUP STORYLINES (MANUSCRIPT)
  group_final4 is assigned directly from WP (1–30) using REGIME_TO_GROUP_4_FINAL.
  This reflects expert-reviewed adjustments (WP16 -> Convective extremes; WP28 -> Cyclonic Atlantic (frontal)).
  For transparency, rule-based outputs are also saved: group_rule6 and group_rule4.

  wf_high      = Q{int(Q_WF_HIGH*100)}(wf_p50)
  wf_low       = Q{int(Q_WF_LOW*100)}(wf_p50)
  tail_high    = Q{int(Q_TAIL_HIGH*100)}(tail_q95_p90_mmph)
  tail_low     = Q{int(Q_TAIL_LOW*100)}(tail_q95_p90_mmph)
  depth_high   = Q{int(Q_DEPTH_HIGH*100)}(depth_p90_mm)
  depth_vhigh  = Q{int(Q_DEPTH_VHIGH*100)}(depth_p90_mm)
  depth_medlow = Q{int(Q_DEPTH_MEDLOW*100)}(depth_p50_mm)

AUDIT WORKBOOK
  {audit_xlsx}

"""
    (out_dir / "README_storylines.txt").write_text(readme, encoding="utf-8")

    print(f"Done. Outputs in:\n  {out_dir}")


if __name__ == "__main__":
    main()
