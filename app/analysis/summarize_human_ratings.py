"""
Summarize human annotation ratings from the annotations CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def summarize_ratings(df, rating_col='accuracy_normalized'):
    """
    Summarize human ratings with descriptive statistics.
    
    Args:
        df: DataFrame with ratings
        rating_col: Column name containing ratings
    """
    ratings = df[rating_col].dropna()
    
    print("\n" + "="*60)
    print("HUMAN RATINGS SUMMARY")
    print("="*60)
    
    print(f"\nTotal annotations: {len(df)}")
    print(f"Valid ratings: {len(ratings)}")
    print(f"Missing ratings: {len(df) - len(ratings)}")
    
    print(f"\nRating Scale Range:")
    print(f"  Minimum: {ratings.min()}")
    print(f"  Maximum: {ratings.max()}")
    print(f"  Range: {ratings.max() - ratings.min()}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Mean: {ratings.mean():.3f}")
    print(f"  Median: {ratings.median():.3f}")
    print(f"  Standard Deviation: {ratings.std():.3f}")
    print(f"  IQR (Q3-Q1): {ratings.quantile(0.75) - ratings.quantile(0.25):.3f}")
    
    print(f"\nDistribution by Rating Value:")
    value_counts = ratings.value_counts().sort_index()
    for value, count in value_counts.items():
        percentage = (count / len(ratings)) * 100
        print(f"  {value:4.0f}: {count:4d} ({percentage:5.2f}%)")
    
    print(f"\nCategorical Breakdown:")
    # Count by preference direction
    positive = (ratings > 0).sum()
    negative = (ratings < 0).sum()
    neutral = (ratings == 0).sum()
    
    print(f"  Prefer S-RAT (positive): {positive:4d} ({positive/len(ratings)*100:5.2f}%)")
    print(f"  Prefer Standard (negative): {negative:4d} ({negative/len(ratings)*100:5.2f}%)")
    print(f"  Equal/Neutral (zero): {neutral:4d} ({neutral/len(ratings)*100:5.2f}%)")
    
    # Strength of preference
    if ratings.abs().max() > 0:
        strong_pref = (ratings.abs() >= 2).sum()
        moderate_pref = (ratings.abs() == 1).sum()
        weak_pref = (ratings.abs() < 1).sum()
        
        print(f"\nStrength of Preference:")
        print(f"  Strong (|rating| >= 2): {strong_pref:4d} ({strong_pref/len(ratings)*100:5.2f}%)")
        print(f"  Moderate (|rating| = 1): {moderate_pref:4d} ({moderate_pref/len(ratings)*100:5.2f}%)")
        print(f"  Weak/Neutral (|rating| < 1): {weak_pref:4d} ({weak_pref/len(ratings)*100:5.2f}%)")
    
    # Shuffling analysis
    if 'is_shuffled' in df.columns:
        print(f"\nShuffling Analysis:")
        shuffled = df['is_shuffled'].value_counts()
        for value, count in shuffled.items():
            print(f"  Shuffled={value}: {count:4d} ({count/len(df)*100:5.2f}%)")
        
        # Compare ratings by shuffling
        if len(shuffled) == 2:
            shuffled_ratings = df[df['is_shuffled'] == True][rating_col].dropna()
            not_shuffled_ratings = df[df['is_shuffled'] == False][rating_col].dropna()
            
            print(f"\n  Mean rating (shuffled): {shuffled_ratings.mean():.3f}")
            print(f"  Mean rating (not shuffled): {not_shuffled_ratings.mean():.3f}")
    
    return {
        'n_total': len(df),
        'n_valid': len(ratings),
        'mean': ratings.mean(),
        'median': ratings.median(),
        'std': ratings.std(),
        'min': ratings.min(),
        'max': ratings.max(),
        'prefer_srat': positive,
        'prefer_standard': negative,
        'neutral': neutral
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize human annotation ratings"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./assets/annotation_results/annotations.csv',
        help='Path to annotations CSV file'
    )
    parser.add_argument(
        '--rating-col',
        type=str,
        default='accuracy_normalized',
        help='Column name for human ratings'
    )
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    if args.rating_col not in df.columns:
        print(f"Error: Rating column '{args.rating_col}' not found")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    # Summarize
    summary = summarize_ratings(df, args.rating_col)
    
    print("\n" + "="*60)
    print("Summary complete!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

