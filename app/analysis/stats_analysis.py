"""
Updated script with data loading from Excel file:
- Each model configuration is in its own sheet.
- Columns: study_id, with_keywords_comet_score (Innovative), without_keywords_comet_score (Standard).
- Script builds df_std and df_inv automatically across all 8 sheets.

Primary analysis:
- Collapse across models: compute mean COMET score per report under Standard and under Innovative.
- Compare using Wilcoxon signed-rank test.

Secondary analysis:
- Per model, compare Standard vs. Innovative with Wilcoxon.
- Apply Holm correction.
- Report effect sizes.

Automatic textual summary is printed at the end.
"""

import itertools
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, norm

# ----------------------------
# Load Excel file
# ----------------------------

file_path = "assets/configuration_comparison/translation_comparison_all.xlsx"  # update path if needed

xls = pd.ExcelFile(file_path)
sheet_names = [s for s in xls.sheet_names if s != "Comparison"]  # exclude summary sheet

all_std = []
all_inv = []

for sheet in sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet)
    # Align columns: Standard = without_keywords, Innovative = with_keywords
    std_col = "without_keywords_comet_score"
    inv_col = "with_keywords_comet_score"

    std_series = df.set_index("study_id")[std_col]
    inv_series = df.set_index("study_id")[inv_col]

    all_std.append(std_series.rename(sheet))
    all_inv.append(inv_series.rename(sheet))

# Combine into two DataFrames aligned by study_id
df_std = pd.concat(all_std, axis=1)
df_inv = pd.concat(all_inv, axis=1)

n_reports, n_models = df_std.shape
model_names = df_std.columns.tolist()

print(f"Loaded data: {n_reports} reports × {n_models} models")

os.makedirs("figs", exist_ok=True)

# ----------------------------
# Primary analysis (collapsed across models)
# ----------------------------

mean_std = df_std.mean(axis=1)
mean_inv = df_inv.mean(axis=1)

W_stat, p_val = wilcoxon(mean_std, mean_inv)
p_clip = max(min(p_val, 1 - 1e-16), 1e-16)
z_abs = norm.ppf(1 - p_clip / 2.0)
r = z_abs / math.sqrt(n_reports)
median_diff = np.median(mean_inv - mean_std)

print("=== Primary analysis (collapsed across models) ===")
print(f"Wilcoxon W={W_stat:.2f}, p={p_val:.3e}, effect size r={r:.3f}")
print(f"Median difference (Innovative - Standard) = {median_diff:.4f}")

# ----------------------------
# Secondary analysis (per model)
# ----------------------------

results = []
pvals = []
W_stats = []
median_diffs = []
for m in model_names:
    W, p = wilcoxon(df_std[m], df_inv[m])
    pvals.append(p)
    W_stats.append(W)
    median_diffs.append(np.median(df_inv[m] - df_std[m]))

# Holm correction
def holm_correction(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj_p = np.empty(m)
    for i, idx in enumerate(order):
        adj_p[idx] = min((m - i) * pvals[idx], 1.0)
    # monotonic adjustment
    for i in range(m-1):
        adj_p[order[i+1]] = max(adj_p[order[i+1]], adj_p[order[i]])
    return adj_p

adj_pvals = holm_correction(pvals)

for m, W, p_raw, p_adj, med in zip(model_names, W_stats, pvals, adj_pvals, median_diffs):
    p_clip = max(min(p_raw, 1 - 1e-16), 1e-16)
    z_abs = norm.ppf(1 - p_clip / 2.0)
    r = z_abs / math.sqrt(n_reports)
    r_signed = r if med >= 0 else -r
    results.append({
        "model": m,
        "W_stat": W,
        "p_raw": p_raw,
        "p_holm": p_adj,
        "median_diff": med,
        "effect_size_r": r_signed
    })

results_df = pd.DataFrame(results)
results_df.to_csv("pairwise_results.csv", index=False)

print("\n=== Secondary analysis (per model) ===")
print(results_df)
print("Saved per-model results to pairwise_results.csv")

# ----------------------------
# Visualization
# ----------------------------

plt.figure(figsize=(10, 6))
plt.boxplot([mean_std, mean_inv], labels=["Standard (avg)", "Innovative (avg)"], showmeans=True)
plt.ylabel("COMET score")
plt.title("Primary comparison: Standard vs Innovative (averaged across models)")
plt.savefig("figs/primary_boxplot.png", dpi=150)
plt.close()

plt.figure(figsize=(12, 6))
plt.boxplot([df_std[m] for m in model_names] + [df_inv[m] for m in model_names],
            labels=[f"Std_{m}" for m in model_names] + [f"Inv_{m}" for m in model_names],
            showmeans=True)
plt.xticks(rotation=45)
plt.ylabel("COMET score")
plt.title("Secondary comparison: Standard vs Innovative per model")
plt.tight_layout()
plt.savefig("figs/secondary_boxplots.png", dpi=150)
plt.close()

print("Saved boxplots to ./figs/")

# ----------------------------
# Automatic textual summary
# ----------------------------

print("\n=== Textual Summary (draft) ===")
if p_val < 0.05:
    print(f"In the primary analysis, the innovative variant achieved significantly different COMET scores compared to the standard variant (Wilcoxon signed-rank test, p={p_val:.3e}, r={r:.3f}). The median improvement was {median_diff:.4f}.")
else:
    print(f"In the primary analysis, the innovative variant did not significantly outperform the standard variant (Wilcoxon signed-rank test, p={p_val:.3e}, r={r:.3f}). The median difference was {median_diff:.4f}.")

signif_models = results_df[results_df["p_holm"] < 0.05]["model"].tolist()
if signif_models:
    print(f"In the secondary analysis, significant improvements were observed for {len(signif_models)} of {n_models} models after Holm correction ({', '.join(signif_models)}).")
else:
    print("In the secondary analysis, no individual model comparisons reached significance after Holm correction, though median differences favored the innovative variant in {} of {} cases.".format((results_df["median_diff"]>0).sum(), n_models))
