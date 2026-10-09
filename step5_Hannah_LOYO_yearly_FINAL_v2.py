#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FINAL reviewer add-on for Step 5: year-specific LOYO performance for Models A-J.

Purpose
-------
This script does NOT change the original model definitions or the LOYO design.
It reads the already-created modelling_dataset_swew.csv and:
  1) repeats the existing leave-one-year-out (LOYO) fitting for Models A-J;
  2) saves AUC and Brier score for each held-out year (2012-2024);
  3) calculates across-year mean, SD and range;
  4) checks that pooled OOF metrics reproduce model_performance_loyo.csv;
  5) calculates paired annual differences for E-C and F-J;
  6) makes a boxplot of annual AUC by model;
  7) writes a publication-ready Table S4a CSV rounded to 3 decimals.

Default paths point to the user's final Step 5 output folder. You can also pass
--input-dir and --output-dir from the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_INPUT_DIR = Path(
    r"C:\Users\earth\OneDrive - University of Reading\PHD 2\Chapters\Paper 2\Codes 2\Analytical procedure\Step 5 Models A-J\outputs_step5_models_2012-2024_May-Sep_20260518_195421"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "reviewer_Hannah_LOYO_by_year"
DEFAULT_SEED = 42


@dataclass
class ModelSpec:
    letter: str
    name: str
    description: str
    features_num: List[str]
    features_cat: List[str]


MODEL_SPECS: List[ModelSpec] = [
    ModelSpec("A", "A_intensity_only", "intensity", ["intensity_q95_mmph"], []),
    ModelSpec("B", "B_depth_only", "depth", ["depth_mm"], []),
    ModelSpec("C", "C_intensity_plus_depth", "intensity + depth", ["intensity_q95_mmph", "depth_mm"], []),
    ModelSpec("D", "D_WP", "WP", [], ["family"]),
    ModelSpec("E", "E_WP_plus_intensity_plus_depth", "WP + intensity + depth", ["intensity_q95_mmph", "depth_mm"], ["family"]),
    ModelSpec("F", "F_WP_plus_wind_UV_SSRD_t2m_soil_API", "WP + wind + UV + SSRD + t2m + soil/API", ["wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C", "swvl1_roll", "p7_mm", "api"], ["family"]),
    ModelSpec("G", "G_full_WP_plus_depth_intensity_plus_wind_UV_SSRD_t2m_soil_API", "WP + depth + intensity + wind + UV + SSRD + t2m + soil/API", ["depth_mm", "intensity_q95_mmph", "wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C", "swvl1_roll", "p7_mm", "api"], ["family"]),
    ModelSpec("H", "H_soil_API_only", "soil / API", ["swvl1_roll", "p7_mm", "api"], []),
    ModelSpec("I", "I_wind_UV_SSRD_t2m_only", "wind + UV + SSRD + t2m", ["wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C"], []),
    ModelSpec("J", "J_wind_UV_SSRD_t2m_plus_soil_API", "wind + UV + SSRD + t2m + soil/API", ["wind_speed", "uvb_Wm2", "ssrd_Wm2", "t2m_C", "swvl1_roll", "p7_mm", "api"], []),
]


def build_pipeline(num_cols: List[str], cat_cols: List[str], seed: int = DEFAULT_SEED) -> Pipeline:
    """Same preprocessing + logistic regression structure as the original Step 5."""
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
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


def fit_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ModelSpec,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one LOYO fold and return raw + sigmoid-calibrated test probabilities."""
    features = spec.features_num + spec.features_cat
    model = build_pipeline(spec.features_num, spec.features_cat, seed=seed)
    model.fit(train[features], train["y_exceed"].astype(int))

    p_test_raw = model.predict_proba(test[features])[:, 1]
    score_train = model.decision_function(train[features])
    score_test = model.decision_function(test[features])

    # Same sigmoid calibration used by the final Step 5 code.
    cal = LogisticRegression(max_iter=2000, solver="lbfgs")
    cal.fit(np.asarray(score_train).reshape(-1, 1), train["y_exceed"].astype(int).to_numpy())
    p_test_cal = cal.predict_proba(np.asarray(score_test).reshape(-1, 1))[:, 1]
    return p_test_raw, p_test_cal


def safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, p))


def run_loyo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-year performance and pooled OOF performance."""
    years = sorted(int(y) for y in df["year"].dropna().unique())
    yearly_rows = []
    pooled_rows = []

    for spec in MODEL_SPECS:
        p_raw_all = pd.Series(np.nan, index=df.index, dtype=float)
        p_cal_all = pd.Series(np.nan, index=df.index, dtype=float)

        for held_year in years:
            train = df[df["year"] != held_year].copy()
            test = df[df["year"] == held_year].copy()
            if test.empty or train.empty:
                continue

            p_raw, p_cal = fit_one_fold(train, test, spec)
            p_raw_all.loc[test.index] = p_raw
            p_cal_all.loc[test.index] = p_cal

            y_test = test["y_exceed"].astype(int).to_numpy()
            auc_raw = safe_auc(y_test, p_raw)
            auc_cal = safe_auc(y_test, p_cal)
            brier_raw = float(brier_score_loss(y_test, p_raw))
            brier_cal = float(brier_score_loss(y_test, p_cal))

            yearly_rows.append({
                "model_letter": spec.letter,
                "model": spec.name,
                "description": spec.description,
                "held_out_year": held_year,
                "n": int(len(test)),
                "n_exceed": int(y_test.sum()),
                "exceedance_rate": float(y_test.mean()),
                "auc_raw": auc_raw,
                "auc_calibrated_sigmoid": auc_cal,
                "brier_raw": brier_raw,
                "brier_calibrated_sigmoid": brier_cal,
            })

        mask = p_raw_all.notna() & p_cal_all.notna()
        y_all = df.loc[mask, "y_exceed"].astype(int).to_numpy()
        pr = p_raw_all.loc[mask].to_numpy()
        pc = p_cal_all.loc[mask].to_numpy()

        pooled_rows.append({
            "model_letter": spec.letter,
            "model": spec.name,
            "description": spec.description,
            "n": int(mask.sum()),
            "pooled_auc_raw": safe_auc(y_all, pr),
            "pooled_auc_calibrated_sigmoid": safe_auc(y_all, pc),
            "pooled_brier_raw": float(brier_score_loss(y_all, pr)),
            "pooled_brier_calibrated_sigmoid": float(brier_score_loss(y_all, pc)),
        })

    return pd.DataFrame(yearly_rows), pd.DataFrame(pooled_rows)


def summarise_yearly(yearly: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    summary = (
        yearly.groupby(["model_letter", "model", "description"], as_index=False)
        .agg(
            n_years=("held_out_year", "nunique"),
            annual_auc_mean=("auc_raw", "mean"),
            annual_auc_sd=("auc_raw", "std"),
            annual_auc_min=("auc_raw", "min"),
            annual_auc_max=("auc_raw", "max"),
            annual_brier_cal_mean=("brier_calibrated_sigmoid", "mean"),
            annual_brier_cal_sd=("brier_calibrated_sigmoid", "std"),
            annual_brier_cal_min=("brier_calibrated_sigmoid", "min"),
            annual_brier_cal_max=("brier_calibrated_sigmoid", "max"),
        )
    )
    out = summary.merge(pooled, on=["model_letter", "model", "description"], how="left")
    order = {s.letter: i for i, s in enumerate(MODEL_SPECS)}
    out["_order"] = out["model_letter"].map(order)
    out = out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return out


def paired_wp_differences(yearly: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annual paired contrasts. Positive delta means WP model is better."""
    pairs = [
        ("E", "C", "E-C", "WP added to intensity + depth"),
        ("F", "J", "F-J", "WP added to meteorology + soil/API"),
    ]
    rows = []
    for wp_letter, base_letter, label, description in pairs:
        wp = yearly[yearly["model_letter"] == wp_letter].set_index("held_out_year")
        base = yearly[yearly["model_letter"] == base_letter].set_index("held_out_year")
        common = sorted(set(wp.index) & set(base.index))
        for year in common:
            rows.append({
                "comparison": label,
                "description": description,
                "held_out_year": int(year),
                "delta_auc_wp_minus_no_wp": float(wp.loc[year, "auc_raw"] - base.loc[year, "auc_raw"]),
                # Lower Brier is better; positive value therefore means WP improved Brier.
                "delta_brier_no_wp_minus_wp": float(base.loc[year, "brier_calibrated_sigmoid"] - wp.loc[year, "brier_calibrated_sigmoid"]),
            })

    by_year = pd.DataFrame(rows)
    summary_rows = []
    for comparison, g in by_year.groupby("comparison"):
        summary_rows.append({
            "comparison": comparison,
            "description": g["description"].iloc[0],
            "n_years": int(len(g)),
            "mean_delta_auc": float(g["delta_auc_wp_minus_no_wp"].mean()),
            "sd_delta_auc": float(g["delta_auc_wp_minus_no_wp"].std(ddof=1)),
            "min_delta_auc": float(g["delta_auc_wp_minus_no_wp"].min()),
            "max_delta_auc": float(g["delta_auc_wp_minus_no_wp"].max()),
            "years_auc_improved": int((g["delta_auc_wp_minus_no_wp"] > 0).sum()),
            "mean_delta_brier_improvement": float(g["delta_brier_no_wp_minus_wp"].mean()),
            "sd_delta_brier_improvement": float(g["delta_brier_no_wp_minus_wp"].std(ddof=1)),
            "years_brier_improved": int((g["delta_brier_no_wp_minus_wp"] > 0).sum()),
        })
    return by_year, pd.DataFrame(summary_rows)


def compare_with_original(pooled: pd.DataFrame, original_path: Path) -> pd.DataFrame:
    """QC: compare reproduced pooled metrics with original model_performance_loyo.csv."""
    if not original_path.exists():
        return pd.DataFrame()
    orig = pd.read_csv(original_path)
    keep = orig.rename(columns={
        "auc": "original_auc",
        "brier_raw": "original_brier_raw",
        "brier_cal_sigmoid": "original_brier_cal_sigmoid",
    })
    q = pooled.merge(keep, on="model", how="left")
    q["diff_auc"] = q["pooled_auc_raw"] - q["original_auc"]
    q["diff_brier_raw"] = q["pooled_brier_raw"] - q["original_brier_raw"]
    q["diff_brier_cal_sigmoid"] = q["pooled_brier_calibrated_sigmoid"] - q["original_brier_cal_sigmoid"]
    return q


def make_table_s4a(summary: pd.DataFrame, original_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build reviewer-ready tables. Where the original pooled output exists, retain those
    exact pooled AUC/Brier values so the table stays consistent with the existing ROC figure.
    Annual mean/SD/range come from the new year-specific LOYO calculation.
    """
    s = summary.copy()
    if original_path.exists():
        orig = pd.read_csv(original_path).rename(columns={
            "auc": "original_pooled_auc",
            "brier_cal_sigmoid": "original_pooled_brier_cal",
        })[["model", "original_pooled_auc", "original_pooled_brier_cal"]]
        s = s.merge(orig, on="model", how="left")
        s["table_pooled_auc"] = s["original_pooled_auc"].fillna(s["pooled_auc_raw"])
        s["table_pooled_brier"] = s["original_pooled_brier_cal"].fillna(s["pooled_brier_calibrated_sigmoid"])
    else:
        s["table_pooled_auc"] = s["pooled_auc_raw"]
        s["table_pooled_brier"] = s["pooled_brier_calibrated_sigmoid"]

    numeric = pd.DataFrame({
        "Model": s["model_letter"],
        "Predictors": s["description"],
        "LOYO years": s["n_years"].astype(int),
        "Pooled AUC": s["table_pooled_auc"],
        "Annual AUC mean": s["annual_auc_mean"],
        "Annual AUC SD": s["annual_auc_sd"],
        "Annual AUC min": s["annual_auc_min"],
        "Annual AUC max": s["annual_auc_max"],
        "Pooled calibrated Brier": s["table_pooled_brier"],
        "Annual calibrated Brier mean": s["annual_brier_cal_mean"],
        "Annual calibrated Brier SD": s["annual_brier_cal_sd"],
    })
    for c in numeric.columns[3:]:
        numeric[c] = numeric[c].round(3)

    copy_ready = pd.DataFrame({
        "Model": s["model_letter"],
        "Predictors": s["description"],
        "Pooled AUC": s["table_pooled_auc"].map(lambda x: f"{x:.3f}"),
        "Annual AUC mean ± SD": s.apply(lambda r: f"{r['annual_auc_mean']:.3f} ± {r['annual_auc_sd']:.3f}", axis=1),
        "Annual AUC range": s.apply(lambda r: f"{r['annual_auc_min']:.3f}–{r['annual_auc_max']:.3f}", axis=1),
        "Pooled calibrated Brier": s["table_pooled_brier"].map(lambda x: f"{x:.3f}"),
        "Annual calibrated Brier mean ± SD": s.apply(lambda r: f"{r['annual_brier_cal_mean']:.3f} ± {r['annual_brier_cal_sd']:.3f}", axis=1),
    })
    return numeric, copy_ready


def plot_auc_boxplot(yearly: pd.DataFrame, pooled: pd.DataFrame, out_path: Path) -> None:
    letters = [s.letter for s in MODEL_SPECS]
    data = [yearly.loc[yearly["model_letter"] == letter, "auc_raw"].dropna().to_numpy() for letter in letters]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Matplotlib-version-safe boxplot labelling (works on older versions too).
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(letters) + 1))
    ax.set_xticklabels(letters)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Model")
    ax.set_ylabel("AUC in held-out year")
    ax.set_title("Year-to-year LOYO discrimination by model (2012–2024)")
    ax.text(
        0.01, 0.01,
        "Boxes show annual LOYO AUC distributions; dashed line = chance (AUC 0.5).",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dataset", default=None, help="Optional explicit modelling_dataset_swew.csv path")
    p.add_argument("--original-performance", default=None, help="Optional explicit model_performance_loyo.csv path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    dataset_path = Path(args.dataset) if args.dataset else input_dir / "modelling_dataset_swew.csv"
    original_path = Path(args.original_performance) if args.original_performance else input_dir / "model_performance_loyo.csv"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "reviewer_Hannah_LOYO_by_year"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Dataset: {dataset_path}")
    print(f"[INFO] Original pooled metrics: {original_path}")
    print(f"[INFO] Output: {output_dir}")

    df = pd.read_csv(dataset_path)
    required = {"year", "y_exceed"}
    for s in MODEL_SPECS:
        required.update(s.features_num)
        required.update(s.features_cat)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].between(2012, 2024)].copy()
    df["y_exceed"] = pd.to_numeric(df["y_exceed"], errors="coerce")
    df = df[df["y_exceed"].isin([0, 1])].copy()

    yearly, pooled = run_loyo(df)
    summary = summarise_yearly(yearly, pooled)
    pair_year, pair_summary = paired_wp_differences(yearly)
    qc = compare_with_original(pooled, original_path)
    table_s4a, table_s4a_copy = make_table_s4a(summary, original_path)

    yearly.to_csv(output_dir / "loyo_performance_by_year.csv", index=False)
    pooled.to_csv(output_dir / "loyo_performance_pooled_reproduced.csv", index=False)
    summary.to_csv(output_dir / "loyo_performance_summary.csv", index=False)
    pair_year.to_csv(output_dir / "loyo_wp_pair_differences_by_year.csv", index=False)
    pair_summary.to_csv(output_dir / "loyo_wp_pair_differences_summary.csv", index=False)
    table_s4a.to_csv(output_dir / "Table_S4a_reviewer_ready.csv", index=False)
    table_s4a_copy.to_csv(output_dir / "Table_S4a_copy_ready.csv", index=False)
    if not qc.empty:
        qc.to_csv(output_dir / "QC_compare_original_pooled_metrics.csv", index=False)

    plot_auc_boxplot(yearly, pooled, output_dir / "Figure_LOYO_AUC_by_year_boxplot.png")

    # Human-readable summary for manuscript revision.
    with (output_dir / "README_reviewer_outputs.txt").open("w", encoding="utf-8") as f:
        f.write("Reviewer LOYO outputs for Hannah comments\n")
        f.write("==========================================\n\n")
        f.write("1. loyo_performance_by_year.csv: AUC/Brier for each held-out year and model.\n")
        f.write("2. loyo_performance_summary.csv: pooled metrics plus annual mean, SD and range.\n")
        f.write("3. Table_S4a_reviewer_ready.csv: numeric rounded table for Supplement S4a.\n")
        f.write("4. Table_S4a_copy_ready.csv: copy-ready mean ± SD / range table.\n")
        f.write("5. Figure_LOYO_AUC_by_year_boxplot.png: annual AUC distribution by model.\n")
        f.write("6. loyo_wp_pair_differences_*: E-C and F-J annual paired differences.\n")
        f.write("7. QC_compare_original_pooled_metrics.csv: checks reproduced pooled values against the original file.\n\n")
        f.write("Interpretation note: positive delta_auc_wp_minus_no_wp means adding WP improved AUC.\n")
        f.write("Positive delta_brier_no_wp_minus_wp means adding WP reduced (improved) calibrated Brier score.\n")

    print("\n[SUMMARY TABLE]")
    print(table_s4a.to_string(index=False))
    print("\n[PAIRED WP CONTRASTS]")
    print(pair_summary.to_string(index=False))
    if not qc.empty:
        print("\n[QC max absolute differences vs original pooled output]")
        for c in ["diff_auc", "diff_brier_raw", "diff_brier_cal_sigmoid"]:
            if c in qc:
                print(f"  {c}: {qc[c].abs().max():.12g}")
    print(f"\n[DONE] Wrote reviewer outputs to: {output_dir}")


if __name__ == "__main__":
    main()
