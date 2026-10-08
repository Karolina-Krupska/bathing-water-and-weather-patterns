# Prevalence (percent) heatmap for May–September weather regimes
# Rows: months (May–September)
# Cols: regimes 1–30
# Values: share (%) of days in that month falling in each regime
# Restricted to 2012–2024 and saved as "figure 2012 - 2024.png"

import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- SETTINGS ----------------
FILE = r"C:\Users\earth\OneDrive - University of Reading\Python codes objective 2\Prevelence of weather patterns\classifications_30regimes(21) (4).csv"
MJJAS = [5, 6, 7, 8, 9]
YEAR_MIN = 2012
YEAR_MAX = 2024

# ------------- Robust reader --------------
def read_regime_calendar(path):
    # Peek at first ~200 lines to find header row & delimiter
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = []
        for _ in range(200):
            try:
                head.append(next(f))
            except StopIteration:
                break
    header_idx = 0
    for i, line in enumerate(head):
        low = line.lower()
        if "regime" in low and ("date" in low or ("year" in low and "month" in low and "day" in low)):
            header_idx = i
            break
    sample = "".join(head[header_idx:header_idx+20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(path, sep=sep, engine="python", header=header_idx,
                       encoding="utf-8-sig", skip_blank_lines=True)

df = read_regime_calendar(FILE)
df.columns = [c.strip().lower() for c in df.columns]

# Build a datetime column
if "date" in df.columns:
    dates = pd.to_datetime(df["date"], errors="coerce")
else:
    for col in ("year","month","day","regime"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    dates = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]), errors="coerce")

# Clean & derive fields
df["regime"] = pd.to_numeric(df["regime"], errors="coerce").astype("Int64")
df = df.assign(date=dates).dropna(subset=["date","regime"]).copy()
df["regime"] = df["regime"].astype(int)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Apply year filtering: 2012–2024
df = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)]
title_span = "2012–2024"  # fixed title span per your request

# Keep May–September only
df = df[df["month"].isin(MJJAS)]

# ---------- Build prevalence table ----------
# Counts per month × regime (summed over selected years)
counts = (
    df.groupby(["month","regime"])
      .size()
      .reindex(pd.MultiIndex.from_product([MJJAS, range(1,31)], names=["month","regime"]), fill_value=0)
      .reset_index(name="day_count")
)

# Pivot to months × regimes
month_names = {5:"May", 6:"June", 7:"July", 8:"August", 9:"September"}
counts["month_name"] = counts["month"].map(month_names)
pivot_counts = counts.pivot(index="month_name", columns="regime", values="day_count").loc[
    ["May","June","July","August","September"]
]

# Row-wise prevalence (% of month)
row_totals = pivot_counts.sum(axis=1).replace(0, np.nan)
pivot_pct = (pivot_counts.div(row_totals, axis=0) * 100).round(1)

# ---------- Plot heatmap (percentages) ----------
fig = plt.figure(figsize=(16, 4.5))
ax = plt.gca()
im = ax.imshow(pivot_pct.values, aspect="auto")  # default colormap (viridis)

# Ticks & labels
ax.set_xticks(np.arange(pivot_pct.shape[1]))
ax.set_xticklabels([str(c) for c in pivot_pct.columns], rotation=0)
ax.set_yticks(np.arange(pivot_pct.shape[0]))
ax.set_yticklabels(list(pivot_pct.index))

# Dashed gridlines
ax.set_xticks(np.arange(-.5, pivot_pct.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-.5, pivot_pct.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linestyle="--", linewidth=0.6, alpha=0.6)
ax.tick_params(which="minor", bottom=False, left=False)

ax.set_xlabel("Weather regime (1–30)")
ax.set_ylabel("Month")
ax.set_title(f"Regime prevalence on May–September days ({title_span})")

# Annotate percentages (hide tiny values)
for i in range(pivot_pct.shape[0]):
    for j in range(pivot_pct.shape[1]):
        v = pivot_pct.values[i, j]
        if not np.isnan(v) and v >= 0.5:
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=8, color="black")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Percent of month (%)")

plt.tight_layout()
out_path = Path(FILE).parent / "figure 2012 - 2024.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print("Saved prevalence heatmap to:", out_path)
plt.show()
