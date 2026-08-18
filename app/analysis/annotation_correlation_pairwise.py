"""
Pairwise correlation analysis between human scores and multiple evaluation metrics.

This script calculates Spearman correlation between human preferences and:
1. COMET score differences
2. Another evaluation score (0-100 scale)

Creates separate visualizations for each correlation.
"""

import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def calculate_correlation(x, y, label_x="X", label_y="Y"):
    """
    Calculate Spearman correlation between two variables.
    
    Args:
        x: First variable
        y: Second variable
        label_x: Label for x variable (for display)
        label_y: Label for y variable (for display)
        
    Returns:
        Tuple of (rho, pval, n_valid)
    """
    # Remove NaN values
    valid_mask = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        print(f"Warning: Insufficient data for correlation between {label_x} and {label_y}")
        return None, None, 0
    
    rho, pval = spearmanr(x_clean, y_clean)
    return rho, pval, len(x_clean)


def create_correlation_plot(
    df, 
    x_col, 
    y_col, 
    x_label, 
    y_label, 
    title,
    output_path,
    zero_lines=True
):
    """
    Create a joint plot with correlation analysis.
    
    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        output_path: Path to save the plot
        zero_lines: Whether to add zero reference lines
    """
    # Calculate correlation
    rho, pval, n = calculate_correlation(df[x_col], df[y_col], x_label, y_label)
    
    if rho is None:
        print(f"Skipping plot for {x_col} vs {y_col} due to insufficient data")
        return
    
    print(f"\n{title}")
    print(f"Spearman's ρ = {rho:.3f}, p = {pval:.4f}, n = {n}")
    
    # Create joint plot
    sns.set(style="whitegrid", font_scale=1.1)
    
    g = sns.jointplot(
        data=df,
        x=x_col,
        y=y_col,
        kind="reg",              # Adds regression trendline
        scatter_kws={'alpha': 0.6, 's': 50},
        line_kws={'color': 'red', 'lw': 2},
        height=6
    )
    
    # Add correlation annotation to title
    plt.suptitle(
        f"{title}\nSpearman's ρ = {rho:.3f}, p = {pval:.4f}",
        fontsize=13,
        y=1.02
    )
    
    # Axis labels
    g.set_axis_labels(x_label, y_label)
    
    # Add zero reference lines if requested
    if zero_lines:
        ax = g.ax_joint
        ax.axvline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def create_combined_plot(
    df,
    comet_col,
    score_100_col,
    human_col,
    comet_label,
    score_100_label,
    human_label,
    output_path
):
    """
    Create a combined figure with two subplots side by side.
    
    Args:
        df: DataFrame with data
        comet_col: Column name for COMET scores
        score_100_col: Column name for 0-100 scale scores
        human_col: Column name for human scores
        comet_label: Label for COMET scores
        score_100_label: Label for 0-100 scale scores
        human_label: Label for human scores
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set(style="whitegrid", font_scale=1.0)
    
    # Plot 1: COMET correlation
    rho_comet, pval_comet, n_comet = calculate_correlation(
        df[comet_col], df[human_col], comet_label, human_label
    )
    
    if rho_comet is not None:
        ax1 = axes[0]
        sns.regplot(
            data=df,
            x=comet_col,
            y=human_col,
            ax=ax1,
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red', 'lw': 2}
        )
        ax1.set_xlabel(comet_label)
        ax1.set_ylabel(human_label)
        ax1.set_title(f"COMET Correlation\nρ = {rho_comet:.3f}, p = {pval_comet:.4f}")
        ax1.axvline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax1.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: 0-100 score correlation
    rho_score, pval_score, n_score = calculate_correlation(
        df[score_100_col], df[human_col], score_100_label, human_label
    )
    
    if rho_score is not None:
        ax2 = axes[1]
        sns.regplot(
            data=df,
            x=score_100_col,
            y=human_col,
            ax=ax2,
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'blue', 'lw': 2}
        )
        ax2.set_xlabel(score_100_label)
        ax2.set_ylabel(human_label)
        ax2.set_title(f"Score (0-100) Correlation\nρ = {rho_score:.3f}, p = {pval_score:.4f}")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pairwise correlations between human scores and evaluation metrics"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./assets/annotation_results/annotations.csv',
        help='Path to annotations CSV file'
    )
    parser.add_argument(
        '--human-col',
        type=str,
        default='accuracy_normalized',
        help='Column name for human preference scores'
    )
    parser.add_argument(
        '--comet-col',
        type=str,
        default='comet-score_delta',
        help='Column name for COMET score differences'
    )
    parser.add_argument(
        '--score-100-col',
        type=str,
        default='gemba',
        help='Column name for the 0-100 scale evaluation score (default: gemba)'
    )
    parser.add_argument(
        '--comet-label',
        type=str,
        default='ΔCOMET (S-RAT – Standard Translation)',
        help='Label for COMET scores in plots'
    )
    parser.add_argument(
        '--score-100-label',
        type=str,
        default='Evaluation Score (0-100)',
        help='Label for 0-100 scale scores in plots'
    )
    parser.add_argument(
        '--human-label',
        type=str,
        default='Human Preference (normalized, + = -K preferred)',
        help='Label for human scores in plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./assets/annotation_results',
        help='Directory to save output plots'
    )
    parser.add_argument(
        '--combined',
        action='store_true',
        help='Create a combined plot with both correlations side by side'
    )
    parser.add_argument(
        '--no-zero-lines',
        action='store_true',
        help='Do not add zero reference lines to plots'
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Check required columns
    required_cols = [args.human_col, args.comet_col, args.score_100_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    output_dir = Path(args.output_dir)
    
    # Calculate and print correlations
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # COMET correlation
    rho_comet, pval_comet, n_comet = calculate_correlation(
        df[args.comet_col], 
        df[args.human_col],
        args.comet_label,
        args.human_label
    )
    
    if rho_comet is not None:
        print(f"\n1. Human vs COMET:")
        print(f"   Spearman's ρ = {rho_comet:.3f}")
        print(f"   p-value = {pval_comet:.4f}")
        print(f"   n = {n_comet}")
    
    # 0-100 score correlation
    rho_score, pval_score, n_score = calculate_correlation(
        df[args.score_100_col],
        df[args.human_col],
        args.score_100_label,
        args.human_label
    )
    
    if rho_score is not None:
        print(f"\n2. Human vs Score (0-100):")
        print(f"   Spearman's ρ = {rho_score:.3f}")
        print(f"   p-value = {pval_score:.4f}")
        print(f"   n = {n_score}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    if args.combined:
        # Create combined plot
        combined_path = output_dir / "correlation_pairwise_combined.png"
        create_combined_plot(
            df,
            args.comet_col,
            args.score_100_col,
            args.human_col,
            args.comet_label,
            args.score_100_label,
            args.human_label,
            combined_path
        )
    else:
        # Create separate plots
        # COMET plot
        comet_path = output_dir / "correlation_comet.png"
        create_correlation_plot(
            df,
            args.comet_col,
            args.human_col,
            args.comet_label,
            args.human_label,
            "Human vs. COMET Preferences",
            comet_path,
            zero_lines=not args.no_zero_lines
        )
        
        # 0-100 score plot
        score_path = output_dir / "correlation_score_100.png"
        create_correlation_plot(
            df,
            args.score_100_col,
            args.human_col,
            args.score_100_label,
            args.human_label,
            "Human vs. Score (0-100) Preferences",
            score_path,
            zero_lines=False  # No zero line for 0-100 scale
        )
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())

