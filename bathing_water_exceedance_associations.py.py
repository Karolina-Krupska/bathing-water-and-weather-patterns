# BWQ_threshold63_IEonly_samplelevel_TWOWAYCLUSTER_RR_rainfall_process_families_2012_2024.py
#
# Purpose
# -------
# This script reproduces the previous sample-level bathing-water analysis, but
# uses storyline_assignments.csv (group_final4) rather than a hard-coded family map.
#
# It:
#   1. Reads bathing-water chemistry files for 2012-2021, 2022, 2023 and 2024.
#   2. Reads the 30-class daily weather-pattern classification file.
#   3. Reads 30 weather-pattern rainfall-process families from storyline_assignments.csv (group_final4).
#   4. Creates an IE-only advice-against-bathing flag using IE >= 63 CFU/100 mL.
#   5. Runs modified Poisson models with two-way clustered standard errors
#      clustered by site and sampling date.
#   6. Exports observed probabilities, adjusted marginal probabilities, risk ratios,
#      figures and an audit dataset.
#
# Main change from the older code:
#   - The four-family assignment now comes from storyline_assignments.csv, column group_final4.
#   - The output folder is new, so older results are not overwritten.
#   - Output names use rainfall-process family wording.

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, chi2
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

ROOT = Path(
    r"C:\Users\earth\OneDrive - University of Reading\Python codes objective 2\Beyond_rainfall_triggers\4 groups vs WP"
)

BW_2012_2021 = ROOT / "BW chem all data 2012 2021.xlsx"
BW_2022 = ROOT / "BW chem data 2022.xlsx"
BW_2023 = ROOT / "BW chem data 2023.xlsx"
BW_2024 = ROOT / "BW chem data 2024.xlsx"

REGIMES_CSV = ROOT / "classifications_30regimes(21) (4).csv"

# Final audited 30-weather-pattern to 4-family mapping.
# This file must contain either:
#   - regime and group_final4 columns, or
#   - exactly 30 rows in WP01-WP30 order with group_final4.
STORYLINE_ASSIGNMENTS_CSV = ROOT / "storyline_assignments.csv"

MONTHS_KEEP = {5, 6, 7, 8, 9}  # May-Sep
YEAR_MIN = 2012
YEAR_MAX = 2024

# Operational threshold for advice against bathing.
ADVICE_THRESHOLD_CFU = 63.0
ADVICE_THRESHOLD_ON = "ie"  # IE-only

# Fixed baseline for 30-pattern RR model.
BASELINE_REGIME = 25

# New output folder, to avoid confusion with earlier runs.
OUTDIR = (
    ROOT
    / "outputs"
    / "bwq_RR_2012_2024_final4"
)
OUTDIR.mkdir(parents=True, exist_ok=True)


def ensure_outdir() -> None:
    """Create the output directory at the point of writing.

    This is intentionally called immediately before every export, because
    OneDrive/Windows can sometimes delay or remove a newly created folder
    during sync. It prevents FileNotFoundError during figure export.
    """
    OUTDIR.mkdir(parents=True, exist_ok=True)


def save_figure_all_formats(fig, outbase: Path, *, tight: bool = False, dpi: int = 300) -> None:
    """Save a matplotlib figure as PNG, PDF and SVG after ensuring the folder exists.

    Uses intentionally short output paths because Windows/Python can raise
    FileNotFoundError when full paths exceed ~260 characters.
    """
    outbase = Path(outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    if not outbase.parent.exists():
        raise FileNotFoundError(f"Output folder was not created: {outbase.parent}")
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, **kwargs)
    fig.savefig(outbase.with_suffix(".pdf"), **kwargs)
    fig.savefig(outbase.with_suffix(".svg"), **kwargs)


# =============================================================================
# EXPECTED FAMILY ORDER
# =============================================================================

FAMILY_ORDER = [
    "Cyclonic Atlantic (frontal)",
    "Convective extremes",
    "Showery maritime / unsettled",
    "Settled anticyclonic quiet",
]

BASELINE_FAMILY = "Settled anticyclonic quiet"


# =============================================================================
# SYNOPTIC LABELS
# =============================================================================

def synoptic_label_map() -> dict[int, str]:
    return {
        1: "Unbiased northwesterly",
        2: "Cyclonic southwesterly (returning polar maritime)",
        3: "Anticyclonic southwesterly (ridge over N France)",
        4: "Unbiased westerly",
        5: "Unbiased southerly (high over Scandinavia)",
        6: "Anticyclonic Azores High extension towards UK",
        7: "Cyclonic southwesterly (low WNW of Ireland)",
        8: "Cyclonic westerly (low near Shetland)",
        9: "Anticyclonic N-N-easterly (high near Iceland)",
        10: "Anticyclonic W-SW (slight Azores ridge)",
        11: "Cyclonic (low centred over southern UK)",
        12: "Anticyclonic southerly (high over Poland)",
        13: "Anticyclonic northwesterly (high SW of Ireland)",
        14: "Cyclonic N-NW (low near southern Sweden)",
        15: "Unbiased southwesterly; very windy in N Britain",
        16: "Anticyclonic S-SE (high east of Denmark)",
        17: "Anticyclonic E-SE (high over Denmark)",
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
        29: "Cyclonic S-SW (deep low west of Ireland)",
        30: "Cyclonic W-SW (deep low SE of Iceland)",
    }


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def parse_censored_numeric(x) -> float:
    """
    Convert microbiology entries to numeric values.
    - '<10' becomes 5
    - '>1000' becomes 1000
    - blanks become NaN
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()

    if s == "":
        return np.nan

    m = re.match(r"^<\s*(\d+(\.\d+)?)$", s)
    if m:
        return 0.5 * float(m.group(1))

    m = re.match(r"^>\s*(\d+(\.\d+)?)$", s)
    if m:
        return float(m.group(1))

    s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return np.nan


def advice_flag(ecoli: float, ie: float, thr: float = ADVICE_THRESHOLD_CFU, mode: str = ADVICE_THRESHOLD_ON) -> bool:
    """Return True if the chosen microbial indicator reaches the operational threshold."""
    mode = str(mode).lower().strip()

    e_ok = np.isfinite(ecoli) and ecoli >= thr
    i_ok = np.isfinite(ie) and ie >= thr

    if mode == "ecoli":
        return bool(e_ok)

    if mode == "ie":
        return bool(i_ok)

    if mode == "either":
        return bool(e_ok or i_ok)

    raise ValueError("ADVICE_THRESHOLD_ON must be one of: 'ecoli', 'ie', 'either'")


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)

    from math import sqrt

    z = 1.959963984540054
    p = k / n

    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom

    return (max(0.0, centre - half), min(1.0, centre + half))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)

    if len(pvals) == 0:
        return pvals

    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]

    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]

    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)

    return out


def pick_column(df: pd.DataFrame, required_terms: list[str], preferred: Optional[str] = None) -> str:
    """
    Find a column whose name contains all required terms, case-insensitive.
    """
    if preferred and preferred in df.columns:
        return preferred

    lower_map = {str(c).lower(): c for c in df.columns}

    for lc, original in lower_map.items():
        if all(term.lower() in lc for term in required_terms):
            return original

    raise ValueError(
        f"Could not find a column containing terms {required_terms}. "
        f"Available columns: {list(df.columns)}"
    )


# =============================================================================
# READ INPUT FILES
# =============================================================================

def read_bw_excel(path: Path, preferred_sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Read one bathing-water chemistry workbook.
    The code is robust to slight sheet-name changes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Bathing-water file not found:\n{path}")

    xls = pd.ExcelFile(path)

    if preferred_sheet and preferred_sheet in xls.sheet_names:
        sheet = preferred_sheet
    else:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)

    date_col = pick_column(df, ["date"], preferred="Date")
    ecoli_col = pick_column(df, ["e_coli"], preferred="2348 E_coli C-MF (NO/100ml)")
    ie_col = pick_column(df, ["ie"], preferred="3723 IE Conf (CFU/0_1l)")
    site_id_col = pick_column(df, ["SMPT_USER_REFERENCE"], preferred="SMPT_USER_REFERENCE")
    site_name_col = pick_column(df, ["SMPT_SHORT_NAME"], preferred="SMPT_SHORT_NAME")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["ecoli"] = df[ecoli_col].apply(parse_censored_numeric)
    out["ie"] = df[ie_col].apply(parse_censored_numeric)
    out["site_id"] = df[site_id_col].astype(str)
    out["site_name"] = df[site_name_col].astype(str)

    out["month"] = out["Date"].dt.month
    out["year"] = out["Date"].dt.year

    out = out.dropna(subset=["Date"]).copy()

    return out[["site_id", "site_name", "Date", "month", "year", "ecoli", "ie"]]


def read_regimes(path: Path) -> pd.DataFrame:
    """
    Read Met Office 30-regime classification file.
    The code auto-detects the header row containing day/month/year/regime.
    """
    if not path.exists():
        raise FileNotFoundError(f"Weather-pattern classification file not found:\n{path}")

    header_row = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            parts = [p.strip().lower() for p in line.strip().split(",")]
            if {"day", "month", "year", "regime"}.issubset(set(parts)):
                header_row = i
                break

    if header_row is None:
        raise ValueError("Could not locate day/month/year/regime header row in classifications CSV.")

    reg = pd.read_csv(path, skiprows=header_row)

    required = {"day", "month", "year", "regime"}
    missing = required - set(reg.columns)

    if missing:
        raise ValueError(f"Regime file missing columns: {sorted(missing)}")

    reg["Date"] = pd.to_datetime(
        dict(
            year=pd.to_numeric(reg["year"], errors="coerce"),
            month=pd.to_numeric(reg["month"], errors="coerce"),
            day=pd.to_numeric(reg["day"], errors="coerce"),
        ),
        errors="coerce",
    )

    reg["regime"] = pd.to_numeric(reg["regime"], errors="coerce").astype("Int64")

    keep_cols = ["Date", "regime"]
    for col in ["distance", "correlation"]:
        if col in reg.columns:
            keep_cols.append(col)

    reg = reg.dropna(subset=["Date", "regime"]).copy()
    reg["regime"] = reg["regime"].astype(int)

    return reg[keep_cols]


def read_family_assignments() -> pd.DataFrame:
    """
    Read the final audited rainfall-process family assignments from
    storyline_assignments.csv.

    Required:
      - group_final4: final four-family assignment used in the manuscript.

    Preferred:
      - regime: Met Office weather-pattern number, 1--30.
      - synoptic_label: optional descriptive label for the weather pattern.

    If the file has no regime column but has exactly 30 rows, rows are assumed
    to be in WP01--WP30 order.
    """
    if not STORYLINE_ASSIGNMENTS_CSV.exists():
        raise FileNotFoundError(
            "storyline_assignments.csv was not found. Expected:\n"
            f"{STORYLINE_ASSIGNMENTS_CSV}\n\n"
            "This script now uses storyline_assignments.csv as the single source "
            "of truth for the WP01--WP30 to group_final4 mapping."
        )

    fam = pd.read_csv(STORYLINE_ASSIGNMENTS_CSV)
    source = STORYLINE_ASSIGNMENTS_CSV

    # Regime number: use explicit column if available; otherwise assume 30-row WP order.
    if any(str(c).lower().strip() == "regime" for c in fam.columns):
        regime_col = [c for c in fam.columns if str(c).lower().strip() == "regime"][0]
        regimes = pd.to_numeric(fam[regime_col], errors="coerce")
    else:
        if len(fam) != 30:
            raise ValueError(
                "storyline_assignments.csv has no regime column and does not have exactly 30 rows. "
                "Please add a regime column with values 1--30."
            )
        regimes = pd.Series(range(1, 31), index=fam.index)

    # Final four-family assignment: this is the manuscript mapping.
    if "group_final4" not in fam.columns:
        raise ValueError(
            "storyline_assignments.csv must contain the column 'group_final4'. "
            f"Available columns are: {list(fam.columns)}"
        )

    out = pd.DataFrame()
    out["regime"] = pd.to_numeric(regimes, errors="coerce").astype("Int64")
    out["group_4"] = fam["group_final4"].astype(str).str.strip()

    # Use the synoptic labels from storyline_assignments.csv when present, because
    # this keeps the 30-pattern descriptions and the final family mapping together.
    if "synoptic_label" in fam.columns:
        out["synoptic_label_from_assignment"] = fam["synoptic_label"].astype(str).str.strip()

    out = out.dropna(subset=["regime"]).copy()
    out["regime"] = out["regime"].astype(int)
    out = out.drop_duplicates(subset=["regime"], keep="first")

    missing_regimes = sorted(set(range(1, 31)) - set(out["regime"]))
    extra_regimes = sorted(set(out["regime"]) - set(range(1, 31)))

    if missing_regimes or extra_regimes:
        raise ValueError(
            "storyline_assignments.csv must define exactly WP01--WP30.\n"
            f"Missing regimes: {missing_regimes}\n"
            f"Unexpected regimes: {extra_regimes}"
        )

    bad_groups = sorted(set(out["group_4"]) - set(FAMILY_ORDER))
    if bad_groups:
        raise ValueError(
            "storyline_assignments.csv contains group_final4 labels outside the expected four families:\n"
            f"{bad_groups}\n"
            f"Expected: {FAMILY_ORDER}"
        )

    # Enforce WP01--WP30 order in the exported audit table.
    out = out.sort_values("regime").reset_index(drop=True)

    print(f"[input] Final WP-to-family mapping: {source}")
    print("[input] Using column: group_final4")

    keep_cols = ["regime", "group_4"]
    if "synoptic_label_from_assignment" in out.columns:
        keep_cols.append("synoptic_label_from_assignment")

    return out[keep_cols]



def build_df() -> pd.DataFrame:
    """
    Build sample-level analysis dataframe.
    """
    bw = pd.concat(
        [
            read_bw_excel(BW_2012_2021, preferred_sheet="Sheet2"),
            read_bw_excel(BW_2022, preferred_sheet="2022"),
            read_bw_excel(BW_2023, preferred_sheet="2023"),
            read_bw_excel(BW_2024, preferred_sheet="2024"),
        ],
        ignore_index=True,
    )

    # Study scope.
    bw = bw[
        bw["month"].isin(MONTHS_KEEP)
        & (bw["year"] >= YEAR_MIN)
        & (bw["year"] <= YEAR_MAX)
    ].copy()

    reg = read_regimes(REGIMES_CSV)
    fam = read_family_assignments()

    bw = bw.merge(reg, on="Date", how="left")
    bw = bw.merge(fam, on="regime", how="left")

    # Merge audit: this is especially important after adding 2024, because
    # the weather-pattern classification file must also contain 2024 dates.
    merge_audit = (
        bw.assign(
            missing_regime=bw["regime"].isna(),
            missing_family=bw["group_4"].isna(),
        )
        .groupby("year", dropna=False)
        .agg(
            n_samples=("Date", "size"),
            n_missing_regime=("missing_regime", "sum"),
            n_missing_family=("missing_family", "sum"),
        )
        .reset_index()
    )
    merge_audit.to_csv(OUTDIR / "merge_audit_by_year.csv", index=False)

    if merge_audit["n_missing_regime"].sum() > 0:
        print("[warning] Some bathing-water rows did not match a weather-pattern date.")
        print(merge_audit.to_string(index=False))

    if "synoptic_label_from_assignment" in bw.columns:
        bw["synoptic_label"] = bw["synoptic_label_from_assignment"].replace({"nan": np.nan})
        bw["synoptic_label"] = bw["synoptic_label"].fillna(bw["regime"].map(synoptic_label_map()))
    else:
        bw["synoptic_label"] = bw["regime"].map(synoptic_label_map())

    bw["advice_flag"] = bw.apply(
        lambda r: advice_flag(r["ecoli"], r["ie"]),
        axis=1,
    ).astype(int)

    bw = bw.dropna(subset=["regime", "group_4"]).copy()
    bw["regime"] = bw["regime"].astype(int)

    if bw.empty:
        raise ValueError("No rows remain after merging bathing-water data, regimes and family assignments.")

    return bw


# =============================================================================
# TWO-WAY CLUSTERED COVARIANCE
# =============================================================================

def _cluster_meat(score_obs: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """Cluster meat matrix from observation-level scores."""
    k = score_obs.shape[1]
    meat = np.zeros((k, k), dtype=float)

    df = pd.DataFrame(score_obs)
    df["_cluster"] = clusters

    grouped = df.groupby("_cluster", sort=False).sum(numeric_only=True)
    grouped = grouped.drop(columns=["_cluster"], errors="ignore")

    for _, row in grouped.iterrows():
        s = row.to_numpy(dtype=float).reshape(-1, 1)
        meat += s @ s.T

    return meat


def two_way_cluster_cov(res, site_ids: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
    """
    Multiway cluster covariance:
      V = B (M_site + M_date - M_site_date) B

    Uses score contributions from the fitted statsmodels GLM.
    """
    score_obs = res.model.score_obs(res.params)

    bread_df = res.cov_params()
    idx = bread_df.index
    bread = bread_df.to_numpy(dtype=float)

    meat_site = _cluster_meat(score_obs, np.asarray(site_ids).astype(str))

    dday = pd.to_datetime(dates).floor("D").to_numpy()
    meat_date = _cluster_meat(score_obs, dday)

    inter = (pd.Series(np.asarray(site_ids).astype(str)) + "||" + pd.Series(dday.astype(str))).to_numpy()
    meat_inter = _cluster_meat(score_obs, inter)

    core = (meat_site + meat_date - meat_inter).astype(float)
    cov = bread @ core @ bread

    return pd.DataFrame(cov, index=idx, columns=idx)


# =============================================================================
# EFFECT TABLES
# =============================================================================

def rr_table_from_cov(res, cov: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Extract risk ratio and clustered inference for terms starting with prefix.
    """
    params = res.params
    se = np.sqrt(np.diag(cov))
    z = params / se
    p = 2 * (1 - norm.cdf(np.abs(z)))

    out = []

    for i, term in enumerate(params.index):
        if str(term).startswith(prefix):
            b = float(params.iloc[i])
            out.append(
                {
                    "term": term,
                    "RR": float(np.exp(b)),
                    "rr_ci_lo": float(np.exp(b - 1.959963984540054 * se[i])),
                    "rr_ci_hi": float(np.exp(b + 1.959963984540054 * se[i])),
                    "p_two_way_cluster": float(p[i]),
                }
            )

    return pd.DataFrame(out).sort_values("RR", ascending=False)


def marginal_mean_and_ci(res, cov: pd.DataFrame, X: pd.DataFrame) -> tuple[float, float, float]:
    """
    Marginal mean of mu_i = exp(x_i beta), averaged over rows of X.
    CI via delta method using supplied covariance.
    """
    beta = res.params.to_numpy()
    Xn = X.to_numpy()

    eta = Xn @ beta
    mu = np.exp(eta)

    m = float(mu.mean())
    grad = (mu[:, None] * Xn).mean(axis=0)

    var = float(grad @ cov.to_numpy() @ grad)
    se = float(np.sqrt(max(var, 0.0)))

    lo = max(0.0, m - 1.96 * se)
    hi = min(1.0, m + 1.96 * se)

    return m, lo, hi


def global_wald_test(res, cov: pd.DataFrame, prefix: str) -> dict[str, float]:
    """
    Joint Wald test for all model coefficients whose names start with prefix.
    Uses the supplied two-way clustered covariance matrix, so the reported
    statistic matches the clustered inference used for the RR tables.
    """
    terms = [t for t in res.params.index if str(t).startswith(prefix)]
    if not terms:
        return {"chi2": np.nan, "df": 0, "p": np.nan}

    b = res.params.loc[terms].to_numpy(dtype=float)
    V = cov.loc[terms, terms].to_numpy(dtype=float)

    # pinv keeps the test stable if clustered covariance is near-singular.
    stat = float(b.T @ np.linalg.pinv(V) @ b)
    df = int(len(terms))
    p = float(chi2.sf(stat, df))
    return {"chi2": stat, "df": df, "p": p}


def _family_term_name(level: str, baseline: str) -> Optional[str]:
    """Return the treatment-coded parameter name for a family level, if non-baseline."""
    if level == baseline:
        return None
    return f"C(group_4, Treatment(reference='{baseline}'))[T.{level}]"


def family_pairwise_contrasts(res, cov: pd.DataFrame) -> pd.DataFrame:
    """
    Pairwise adjusted RR contrasts among the four rainfall-process families.
    Contrasts are on the log scale and use the two-way clustered covariance.
    """
    rows = []
    params = res.params
    terms = list(params.index)

    for i, a in enumerate(FAMILY_ORDER):
        for b in FAMILY_ORDER[i + 1:]:
            c = np.zeros(len(terms), dtype=float)

            ta = _family_term_name(a, BASELINE_FAMILY)
            tb = _family_term_name(b, BASELINE_FAMILY)

            if ta is not None and ta in terms:
                c[terms.index(ta)] += 1.0
            if tb is not None and tb in terms:
                c[terms.index(tb)] -= 1.0

            est = float(c @ params.to_numpy(dtype=float))
            var = float(c @ cov.to_numpy(dtype=float) @ c)
            se = float(np.sqrt(max(var, 0.0)))
            z = est / se if se > 0 else np.nan
            p = float(2 * (1 - norm.cdf(abs(z)))) if np.isfinite(z) else np.nan

            rows.append(
                {
                    "contrast": f"{a} vs {b}",
                    "level_a": a,
                    "level_b": b,
                    "RR_a_vs_b": float(np.exp(est)),
                    "rr_ci_lo": float(np.exp(est - 1.959963984540054 * se)),
                    "rr_ci_hi": float(np.exp(est + 1.959963984540054 * se)),
                    "p_two_way_cluster": p,
                }
            )

    out = pd.DataFrame(rows)
    out["q_fdr_bh"] = bh_fdr(out["p_two_way_cluster"].to_numpy(dtype=float))
    return out


def plot_family_probability_with_stats(
    prob_df: pd.DataFrame,
    rr_df: pd.DataFrame,
    wald: dict[str, float],
    outbase: Path,
):
    """
    Main-text style family probability figure with RR/p-value annotations.
    This keeps the probability scale in the main panel while giving the
    significance information requested by reviewers directly on the figure.
    """
    order = [
        "Cyclonic Atlantic (frontal)",
        "Showery maritime / unsettled",
        "Convective extremes",
        BASELINE_FAMILY,
    ]

    d = prob_df.copy().set_index("group").loc[order].reset_index()
    rr_lookup = rr_df.set_index("label") if not rr_df.empty else pd.DataFrame()

    y = np.arange(len(d)) * 1.45
    fig = plt.figure(figsize=(12.8, 6.0))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.6, 2.0], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_txt.axis("off")

    x = d["p"].to_numpy(dtype=float)
    lo = d["lo"].to_numpy(dtype=float)
    hi = d["hi"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=4, markersize=7)

    # Label rows; add * where different from settled at p<0.05.
    ylabels = []
    for _, row in d.iterrows():
        g = row["group"]
        suffix = ""
        if g != BASELINE_FAMILY and g in rr_lookup.index:
            pval = float(rr_lookup.loc[g, "p_two_way_cluster"])
            if pval < 0.05:
                suffix = " *"
        ylabels.append(f"{g} (n={int(row['n'])}){suffix}")

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("Adjusted predicted probability of advice against bathing (IE >= 63 CFU/100 mL)")
    ax.set_title("Advice against bathing probability by rainfall-process family (adjusted)")

    xmin = float(np.nanmin(lo))
    xmax = float(np.nanmax(hi))
    span = max(xmax - xmin, 1e-6)
    ax.set_xlim(max(0, xmin - 0.10 * span), xmax + 0.18 * span)

    for i, val in enumerate(x):
        star = ""
        g = d.loc[i, "group"]
        if g != BASELINE_FAMILY and g in rr_lookup.index:
            if float(rr_lookup.loc[g, "p_two_way_cluster"]) < 0.05:
                star = " *"
        ax.annotate(
            f"{val * 100:.1f}%{star}",
            xy=(val, y[i]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.2),
            clip_on=False,
        )

    for i, row in d.iterrows():
        g = row["group"]
        if g == BASELINE_FAMILY:
            txt = "Reference group"
        elif g in rr_lookup.index:
            r = rr_lookup.loc[g]
            pval = float(r["p_two_way_cluster"])
            if pval < 0.001:
                ptxt = "p<0.001"
            else:
                ptxt = f"p={pval:.3f}"
            ns = ", ns" if pval >= 0.05 else ""
            txt = f"vs settled: RR {r['RR']:.2f} ({r['rr_ci_lo']:.2f}-{r['rr_ci_hi']:.2f}), {ptxt}{ns}"
        else:
            txt = ""
        ax_txt.text(0.0, y[i], txt, va="center", fontsize=10)

    wtxt = f"Global Wald test: chi2({wald['df']})={wald['chi2']:.1f}, p={wald['p']:.2e}"
    fig.text(0.39, 0.055, wtxt, fontsize=10, ha="left")
    fig.text(0.39, 0.025, "* p<0.05 vs settled anticyclonic quiet; ns = not significant.", fontsize=9, ha="left", style="italic")

    fig.subplots_adjust(left=0.32, right=0.98, top=0.88, bottom=0.17)
    save_figure_all_formats(fig, outbase, tight=True, dpi=300)
    plt.close(fig)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_rr_forest(rr_df: pd.DataFrame, labels: list[str], outbase: Path, title: str):
    """
    Forest plot for risk ratios on log scale, including reference row.
    """
    d = rr_df.copy()
    d["label"] = pd.Categorical(d["label"], categories=labels, ordered=True)
    d = d.sort_values("label").reset_index(drop=True)

    y = np.arange(len(d)) * 1.45
    height = max(4.8, 0.34 * len(d) + 1.2)

    fig = plt.figure(figsize=(10.5, height))
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3.2, 1.5], wspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1], sharey=ax)
    ax_txt.axis("off")

    is_ref = d["is_ref"].to_numpy(dtype=bool)
    non_ref = ~is_ref

    if non_ref.any():
        x = d.loc[non_ref, "RR"].to_numpy(dtype=float)
        lo = d.loc[non_ref, "lo"].to_numpy(dtype=float)
        hi = d.loc[non_ref, "hi"].to_numpy(dtype=float)
        xerr = np.vstack([x - lo, hi - x])
        ax.errorbar(x, y[non_ref], xerr=xerr, fmt="o", capsize=3)

    if is_ref.any():
        ax.plot([1.0], y[is_ref], marker="o")

    ax.axvline(1.0, linestyle="--")
    ax.set_xscale("log")

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])
    ax.set_xlabel("Risk ratio (RR)")
    ax.set_title(title)
    ax.invert_yaxis()

    for i in range(len(d)):
        row = d.iloc[i]

        if bool(row["is_ref"]):
            txt = "RR 1.00"
        else:
            txt = f"RR {row['RR']:.2f} ({row['lo']:.2f}, {row['hi']:.2f})"

        ax_txt.text(0.0, y[i], txt, va="center", fontsize=9)

    fig.subplots_adjust(left=0.36, right=0.98, top=0.93, bottom=0.16)

    save_figure_all_formats(fig, outbase, tight=False, dpi=300)

    plt.close(fig)


def plot_probability_dotwhisker(prob_df: pd.DataFrame, outbase: Path, title: str, xlabel: str):
    """
    Dot-and-whisker probability plot with 95% CI.
    """
    d = prob_df.copy().sort_values("p", ascending=False).reset_index(drop=True)

    y_step = 1.75
    y = np.arange(len(d)) * y_step

    height = max(5.2, 0.42 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(10.8, height))

    x = d["p"].to_numpy(dtype=float)
    lo = d["lo"].to_numpy(dtype=float)
    hi = d["hi"].to_numpy(dtype=float)

    xerr = np.vstack([x - lo, hi - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"])

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    ax.invert_yaxis()

    ypad = 0.85 * y_step
    ax.set_ylim(y.max() + ypad, -ypad)

    xmin = float(np.nanmin(lo))
    xmax = float(np.nanmax(hi))
    x_range = (xmax - xmin) if xmax > xmin else max(xmax, 1e-6)

    pad_left = 0.08 * x_range
    pad_right = 0.30 * x_range

    ax.set_xlim(max(0.0, xmin - pad_left), xmax + pad_right)

    for i, val in enumerate(x):
        if val >= xmin + 0.92 * x_range:
            dx, ha = -6, "right"
        elif val <= xmin + 0.08 * x_range:
            dx, ha = 6, "left"
        else:
            dx, ha = 0, "center"

        ax.annotate(
            f"{val * 100:.1f}%",
            xy=(val, y[i]),
            xytext=(dx, 14),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=9,
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.25, alpha=0.95),
            clip_on=False,
        )

    max_lab_len = max(len(str(s)) for s in d["label"])
    left = min(0.52, max(0.30, 0.18 + 0.006 * max_lab_len))

    fig.subplots_adjust(left=left, right=0.98, top=0.92, bottom=0.16)

    save_figure_all_formats(fig, outbase, tight=True, dpi=300)

    plt.close(fig)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main() -> None:
    ensure_outdir()
    print(f"[output] Results will be written to: {OUTDIR}")
    df = build_df()

    # Audit dataset.
    ensure_outdir()
    df.to_csv(OUTDIR / "sample_level_audit.csv", index=False)

    df_model = df.copy()

    # Ensure family categories have consistent order.
    observed_families = list(pd.Series(df_model["group_4"].dropna().unique()).astype(str))

    missing_family_levels = [g for g in FAMILY_ORDER if g not in observed_families]
    if missing_family_levels:
        raise ValueError(
            "The following expected family levels are not present in the merged dataset:\n"
            f"{missing_family_levels}"
        )

    df_model["group_4"] = pd.Categorical(df_model["group_4"], categories=FAMILY_ORDER, ordered=False)
    df_model["year"] = df_model["year"].astype(int).astype(str)

    # -------------------------------------------------------------------------
    # A) Four rainfall-process families.
    # -------------------------------------------------------------------------

    if BASELINE_FAMILY not in FAMILY_ORDER:
        raise ValueError(f"Baseline family not in FAMILY_ORDER: {BASELINE_FAMILY}")

    formula_g = f"advice_flag ~ C(group_4, Treatment(reference='{BASELINE_FAMILY}')) + C(month) + C(year)"

    glm_g = smf.glm(formula_g, data=df_model, family=sm.families.Poisson())
    res_g = glm_g.fit()

    cov_g = two_way_cluster_cov(
        res_g,
        site_ids=df_model["site_id"].to_numpy(),
        dates=df_model["Date"].to_numpy(),
    )

    rr_g = rr_table_from_cov(res_g, cov_g, prefix="C(group_4")
    rr_g["label"] = rr_g["term"].str.replace(r".*\[T\.", "", regex=True).str.replace(r"\]$", "", regex=True)
    family_global_wald = global_wald_test(res_g, cov_g, prefix="C(group_4")
    family_pairwise = family_pairwise_contrasts(res_g, cov_g)

    labels_g = [
        "Cyclonic Atlantic (frontal)",
        "Showery maritime / unsettled",
        "Convective extremes",
        f"{BASELINE_FAMILY} (ref.)",
    ]

    rr_forest = pd.DataFrame(
        {
            "label": rr_g["label"].tolist() + [f"{BASELINE_FAMILY} (ref.)"],
            "RR": rr_g["RR"].tolist() + [1.0],
            "lo": rr_g["rr_ci_lo"].tolist() + [np.nan],
            "hi": rr_g["rr_ci_hi"].tolist() + [np.nan],
            "is_ref": [False] * len(rr_g) + [True],
        }
    )

    rr_forest = rr_forest.set_index("label").loc[labels_g].reset_index()

    plot_rr_forest(
        rr_forest,
        labels=labels_g,
        outbase=OUTDIR / "family4_RR",
        title="Adjusted RR for advice against bathing by rainfall-process family",
    )

    # Observed family probabilities.
    obs_rows = []

    for g, d in df_model.groupby("group_4", sort=False):
        n = len(d)
        k = int(d["advice_flag"].sum())
        p = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)

        obs_rows.append(
            {
                "group": g,
                "n": n,
                "p": p,
                "lo": lo,
                "hi": hi,
                "label": f"{g} (n={n})",
            }
        )

    obs_g = pd.DataFrame(obs_rows).set_index("group").loc[FAMILY_ORDER].reset_index()

    plot_probability_dotwhisker(
        obs_g,
        outbase=OUTDIR / "family4_prob_obs",
        title="Advice against bathing probability by rainfall-process family (observed)",
        xlabel="Observed probability of advice against bathing (IE >= 63 CFU/100 mL)",
    )

    # Adjusted marginal probabilities by family.
    import patsy

    _, _ = patsy.dmatrices(formula_g, df_model, return_type="dataframe")

    adj_rows = []

    for g in FAMILY_ORDER:
        dtmp = df_model.copy()
        dtmp["group_4"] = g
        dtmp["group_4"] = pd.Categorical(dtmp["group_4"], categories=FAMILY_ORDER, ordered=False)

        _, Xg = patsy.dmatrices(formula_g, dtmp, return_type="dataframe")

        m, lo, hi = marginal_mean_and_ci(res_g, cov_g, Xg)

        n = int(obs_g.loc[obs_g["group"] == g, "n"].iloc[0])

        adj_rows.append(
            {
                "group": g,
                "n": n,
                "p": m,
                "lo": lo,
                "hi": hi,
                "label": f"{g} (n={n})",
            }
        )

    adj_g = pd.DataFrame(adj_rows)

    plot_probability_dotwhisker(
        adj_g,
        outbase=OUTDIR / "family4_adjprob",
        title="Advice against bathing probability by rainfall-process family (adjusted)",
        xlabel="Adjusted predicted probability of advice against bathing (IE >= 63 CFU/100 mL)",
    )

    plot_family_probability_with_stats(
        adj_g,
        rr_g,
        family_global_wald,
        outbase=OUTDIR / "family4_adjprob_stats",
    )

    # -------------------------------------------------------------------------
    # B) 30 weather patterns.
    # -------------------------------------------------------------------------

    baseline_regime = BASELINE_REGIME

    if baseline_regime not in df_model["regime"].unique():
        raise ValueError(f"Requested baseline_regime={baseline_regime} not present in data.")

    formula_r = f"advice_flag ~ C(regime, Treatment(reference={baseline_regime})) + C(month) + C(year)"

    glm_r = smf.glm(formula_r, data=df_model, family=sm.families.Poisson())
    res_r = glm_r.fit()

    cov_r = two_way_cluster_cov(
        res_r,
        site_ids=df_model["site_id"].to_numpy(),
        dates=df_model["Date"].to_numpy(),
    )
    weatherpattern_global_wald = global_wald_test(res_r, cov_r, prefix="C(regime")

    rr_r = rr_table_from_cov(res_r, cov_r, prefix="C(regime")
    rr_r["regime"] = rr_r["term"].str.extract(r"\[T\.(\d+)\]").astype(int)

    n_by_regime = df_model.groupby("regime").size().rename("n")
    rr_r = rr_r.merge(n_by_regime, left_on="regime", right_index=True, how="left")

    rr_r["q_fdr_bh"] = bh_fdr(rr_r["p_two_way_cluster"].to_numpy())

    all_regimes = sorted(df_model["regime"].unique())

    base_label = f"{baseline_regime:02d} (ref.)"

    rr_plot = pd.DataFrame(
        {
            "label": [f"{r:02d}" for r in rr_r["regime"]] + [base_label],
            "RR": rr_r["RR"].tolist() + [1.0],
            "lo": rr_r["rr_ci_lo"].tolist() + [np.nan],
            "hi": rr_r["rr_ci_hi"].tolist() + [np.nan],
            "is_ref": [False] * len(rr_r) + [True],
            "q": rr_r["q_fdr_bh"].tolist() + [np.nan],
            "n": rr_r["n"].tolist() + [int(n_by_regime.loc[baseline_regime])],
        }
    )

    rr_plot["label"] = rr_plot.apply(
        lambda r: f"WP {r['label']} (n={int(r['n'])})"
        if not r["is_ref"]
        else f"WP {baseline_regime:02d} (n={int(r['n'])}) (ref.)",
        axis=1,
    )

    rr_plot = rr_plot.sort_values("RR", ascending=False, na_position="last").reset_index(drop=True)

    rr_plot["label"] = rr_plot.apply(
        lambda r: (r["label"] + " *")
        if (not r["is_ref"] and pd.notna(r["q"]) and r["q"] < 0.05)
        else r["label"],
        axis=1,
    )

    plot_rr_forest(
        rr_plot[["label", "RR", "lo", "hi", "is_ref"]],
        labels=rr_plot["label"].tolist(),
        outbase=OUTDIR / "wp30_RR_ref25",
        title="Adjusted RR for advice against bathing by weather pattern",
    )

    # Observed probabilities by weather pattern.
    obs_rows = []

    for r, d in df_model.groupby("regime", sort=True):
        n = len(d)
        k = int(d["advice_flag"].sum())
        p = k / n if n else np.nan
        lo, hi = wilson_ci(k, n)

        obs_rows.append(
            {
                "regime": r,
                "n": n,
                "p": p,
                "lo": lo,
                "hi": hi,
                "label": f"WP {r:02d} (n={n})",
            }
        )

    obs_r = pd.DataFrame(obs_rows)

    plot_probability_dotwhisker(
        obs_r,
        outbase=OUTDIR / "wp30_prob_obs",
        title="Advice against bathing probability by weather pattern (observed)",
        xlabel="Observed probability of advice against bathing (IE >= 63 CFU/100 mL)",
    )

    # Adjusted marginal probabilities by weather pattern.
    _, _ = patsy.dmatrices(formula_r, df_model, return_type="dataframe")

    adj_rows = []

    for r in all_regimes:
        dtmp = df_model.copy()
        dtmp["regime"] = r
        dtmp["regime"] = pd.Categorical(dtmp["regime"], categories=all_regimes, ordered=True)

        _, Xr = patsy.dmatrices(formula_r, dtmp, return_type="dataframe")

        m, lo, hi = marginal_mean_and_ci(res_r, cov_r, Xr)

        n = int(n_by_regime.loc[r])

        adj_rows.append(
            {
                "regime": r,
                "n": n,
                "p": m,
                "lo": lo,
                "hi": hi,
                "label": f"WP {r:02d} (n={n})",
            }
        )

    adj_r = pd.DataFrame(adj_rows)

    plot_probability_dotwhisker(
        adj_r,
        outbase=OUTDIR / "wp30_adjprob",
        title="Advice against bathing probability by weather pattern (adjusted)",
        xlabel="Adjusted predicted probability of advice against bathing (IE >= 63 CFU/100 mL)",
    )

    # -------------------------------------------------------------------------
    # Save summary workbook.
    # -------------------------------------------------------------------------

    out_xlsx = OUTDIR / "bwq_summary_IE63_RR_2012_2024.xlsx"

    family_map_used = read_family_assignments()

    ensure_outdir()
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="sample_level_analysis_data", index=False)
        family_map_used.to_excel(w, sheet_name="storyline_assignments_used", index=False)

        rr_g.to_excel(w, sheet_name="family4_adjusted_RR", index=False)
        obs_g.to_excel(w, sheet_name="family4_prob_observed", index=False)
        adj_g.to_excel(w, sheet_name="family4_prob_adjusted", index=False)
        pd.DataFrame([family_global_wald]).to_excel(w, sheet_name="family4_global_Wald", index=False)
        family_pairwise.to_excel(w, sheet_name="family4_pairwise_contrasts", index=False)

        rr_r.to_excel(w, sheet_name="weatherpattern30_adjusted_RR", index=False)
        obs_r.to_excel(w, sheet_name="weatherpattern30_prob_observed", index=False)
        adj_r.to_excel(w, sheet_name="weatherpattern30_prob_adjusted", index=False)
        pd.DataFrame([weatherpattern_global_wald]).to_excel(w, sheet_name="weatherpattern30_global_Wald", index=False)

    readme = f"""Bathing-water advice-against-bathing analysis by rainfall-process family and weather pattern

INPUT FOLDER
  {ROOT}

BATHING-WATER FILES
  {BW_2012_2021.name}
  {BW_2022.name}
  {BW_2023.name}
  {BW_2024.name}

WEATHER-PATTERN FILE
  {REGIMES_CSV.name}

RAINFALL-PROCESS FAMILY FILE
  {STORYLINE_ASSIGNMENTS_CSV.name}
  Mapping column: group_final4

OUTPUT FOLDER
  {OUTDIR}

FILTERS
  months: {sorted(MONTHS_KEEP)}
  years: {YEAR_MIN}-{YEAR_MAX}

THRESHOLD
  advice_flag = 1 when IE >= {ADVICE_THRESHOLD_CFU} CFU/100 mL

MODELS
  Modified Poisson regression with log link.
  Two-way clustered sandwich standard errors by site and sampling date.
  Models adjusted for month and year.
  Global Wald tests and pairwise family contrasts are exported using the same two-way clustered covariance.

FOUR FAMILIES
  {chr(10).join("- " + x for x in FAMILY_ORDER)}

BASELINES
  Family model baseline: {BASELINE_FAMILY}
  Weather-pattern model baseline: WP{BASELINE_REGIME:02d}

MAIN WORKBOOK
  {out_xlsx.name}

FIGURES
  Exported as PNG, PDF and SVG.
  Main-text-style family probability figure with RR/p-value annotations:
    family4_adjprob_stats.png
"""

    ensure_outdir()
    (OUTDIR / "README_analysis.txt").write_text(readme, encoding="utf-8")

    print("Done. Outputs written to:")
    print(OUTDIR)
    print("Main workbook:")
    print(out_xlsx)
    print("Rows in analysis dataset:", len(df))
    print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())


if __name__ == "__main__":
    main()
