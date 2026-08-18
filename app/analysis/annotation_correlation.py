import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("./assets/annotation_results/annotations.csv")


# Compute Spearman correlation
rho, pval = spearmanr(df["accuracy_normalized"], df["comet-score_delta"])

print(f"Spearman's ρ = {rho:.3f}, p = {pval:.4f}")

# --- STEP 3: Create a nice joint plot with regression trend ---
sns.set(style="whitegrid", font_scale=1.1)

g = sns.jointplot(
    data=df,
    x="comet-score_delta",
    y="accuracy_normalized",
    kind="reg",              # Adds regression trendline
    scatter_kws={'alpha': 0.6, 's': 50},
    line_kws={'color': 'red', 'lw': 2},
    height=6
)

# --- STEP 4: Add correlation annotation ---
plt.suptitle(
    f"Human vs. COMET-src Preferences\nSpearman’s ρ = {rho:.3f}, p = {pval:.4f}",
    fontsize=13,
    y=1.02
)

# Axis labels
g.set_axis_labels(
    "ΔCOMET (S-RAT – Standard Translation)",
    "Human Preference (normalized, + = -K preferred)"
)

# --- STEP 5: Optional: add zero reference lines ---
ax = g.ax_joint
ax.axvline(0, color='gray', linestyle='--', lw=1)
ax.axhline(0, color='gray', linestyle='--', lw=1)

plt.tight_layout()
plt.show()
