import json
import pandas as pd
from collections import Counter
from statsmodels.stats.proportion import proportion_confint

# --- Load data ---
with open("assets/failed_translation_analysis/failed_translations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# --- Extract top-level counts ---
T = data["total_concepts"]          # total extracted concept instances
U = data["total_failed"]            # total untranslated instances
coverage_obs = 1 - (U / T)

print(f"Total instances: {T}")
print(f"Untranslated instances: {U}")
print(f"Observed coverage: {coverage_obs:.3f} ({coverage_obs*100:.2f}%)")

# --- Confidence interval (95% Wilson) ---
ci_low, ci_high = proportion_confint(count=T-U, nobs=T, alpha=0.05, method="wilson")
print(f"95% CI for coverage: {ci_low:.3f} – {ci_high:.3f}")

# --- Build DataFrame of failed translations ---
failed = pd.DataFrame(data["failed_translations"])

# Add a flag for SNOMED lookup success/failure
failed["snomed_success"] = failed["snomed_lookup"].apply(lambda x: x.get("success") if x else False)
failed["snomed_code"] = failed["snomed_lookup"].apply(lambda x: x.get("snomed_code") if x else None)

# --- Summary by category ---
summary_by_category = failed.groupby("category").size().reset_index(name="count").sort_values("count", ascending=False)
print("\nUntranslated by category:")
print(summary_by_category)

# --- Top untranslated terms (frequency) ---
top_terms = failed["normalized_text"].value_counts().reset_index()
top_terms.columns = ["term", "count"]
print("\nTop untranslated terms:")
print(top_terms.head(10))

# --- Distinct-type coverage ---
D = failed["normalized_text"].nunique() + (T - U)  # crude: distinct untranslated + translated
K = failed["normalized_text"].nunique()            # distinct untranslated types
distinct_coverage = 1 - (K / D)

print(f"\nDistinct-type coverage: {distinct_coverage:.3f} ({distinct_coverage*100:.2f}%)")
print(f"Unique untranslated types: {K}")

# --- Pareto table (top-N) ---
top_n = 20
pareto = top_terms.head(top_n).copy()
pareto["cum_share"] = pareto["count"].cumsum() / pareto["count"].sum()
print(f"\nTop {top_n} untranslated terms (with cumulative share):")
print(pareto)

# --- Optional: export tables for LaTeX or CSV ---
summary_by_category.to_csv("untranslated_by_category.csv", index=False)
top_terms.to_csv("top_untranslated_terms.csv", index=False)
pareto.to_csv("pareto_untranslated_terms.csv", index=False)
