"""Module for creating Excel files to compare translation modes with evaluation scores.

This module creates Excel files with side-by-side comparisons of two translation modes:
- with_keywords: Translation using extracted medical concepts and SNOMED translations
- without_keywords: Direct translation without concept extraction

Key features:
- Random shuffling of translation order to prevent annotation bias
- Tracking column to identify which rows were shuffled
- Single numerical comparison column (-2 to +2 scale) for easy annotation
- Summary and statistics sheets for analysis
- UMLS lookup tracking: Detailed information about which concepts required UMLS lookups
- Enhanced statistics including UMLS lookup success rates and patterns
- COMET evaluation scores: Automatic quality assessment using COMET metrics
- Evaluation comparison: Direct comparison of COMET scores between translation modes

UMLS Tracking Features:
- concepts_requiring_umls: Detailed list of concepts that needed UMLS lookup for SNOMED codes
- num_umls_lookups: Count of concepts requiring UMLS lookup
- UMLS lookup success rates and patterns in statistics
- Timestamps and full context for each UMLS lookup

Evaluation Features:
- with_keywords_comet_score: COMET quality score for with_keywords translations
- without_keywords_comet_score: COMET quality score for without_keywords translations
- comet_score_difference: Direct comparison of quality scores
- evaluation_model: COMET model used for evaluation
- evaluation_timestamp: When the evaluation was performed

Annotation method:
- comparison_score: Numerical value comparing translation_a vs translation_b
  -2: translation_a much worse than translation_b
  -1: translation_a worse than translation_b
   0: translation_a equal to translation_b
  +1: translation_a better than translation_b
  +2: translation_a much better than translation_b

Usage:
    python app/excel_comparison.py [--no-shuffle] [--random-seed 42]
"""

import os
import glob
import random
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
from exceptions import EvaluationError


def find_latest_translation_dataset(base_dir: str = "data/evaluations") -> str:
    """
    Find the latest translation dataset with evaluation scores.
    
    Args:
        base_dir: Base directory to search for evaluation datasets
        
    Returns:
        Path to the latest translation dataset with evaluation
        
    Raises:
        EvaluationError: If no translation datasets with evaluation are found
    """
    if not os.path.exists(base_dir):
        raise EvaluationError(f"Evaluation directory not found: {base_dir}")
    
    # Find all subdirectories in the base directory
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    if not subdirs:
        raise EvaluationError(f"No subdirectories found in {base_dir}")
    
    # Sort subdirectories by name (assuming they are timestamped) to get the latest
    latest_subdir = sorted(subdirs)[-1]
    latest_subdir_path = os.path.join(base_dir, latest_subdir)
    
    # Look for translations_with_evaluation.json in the latest subdirectory
    dataset_path = os.path.join(latest_subdir_path, "translations_with_evaluation.json")
    
    if not os.path.exists(dataset_path):
        raise EvaluationError(f"Dataset file not found: {dataset_path}")
    
    print(f"Found latest translation dataset with evaluation: {dataset_path}")
    return dataset_path


def load_translation_dataset(dataset_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load translation dataset with evaluation scores and metadata.
    
    Args:
        dataset_path: Path to the translation dataset with evaluation (None to use latest)
        
    Returns:
        Tuple of (dataset, metadata)
    """
    if dataset_path is None:
        dataset_path = find_latest_translation_dataset()
    
    # Load the dataset
    print(f"Loading translation dataset with evaluation from: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Load metadata - try to find it in the same directory or parent directories
    metadata = {}
    current_dir = os.path.dirname(dataset_path)
    
    # Look for metadata.json in the same directory
    metadata_path = os.path.join(current_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        # Look for evaluation_summary.json which contains some metadata
        summary_path = os.path.join(current_dir, "evaluation_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                metadata.update({
                    'evaluation_model': summary_data.get('model_name', ''),
                    'evaluation_timestamp': summary_data.get('evaluation_timestamp', ''),
                    'system_score': summary_data.get('system_score', 0),
                    'mean_score': summary_data.get('mean_score', 0)
                })
    
    return dataset, metadata


def create_comparison_dataframe(dataset, shuffle_order: bool = True, random_seed: Optional[int] = None) -> pd.DataFrame:
    """
    Create a DataFrame for comparing translation modes.
    
    Args:
        dataset: The loaded translation dataset
        shuffle_order: Whether to randomly shuffle the order of translation modes
        random_seed: Random seed for reproducible shuffling
        
    Returns:
        DataFrame with side-by-side comparison of translation modes
    """
    # Separate translations by mode
    with_keywords = [item for item in dataset if item.get('translation_mode') == 'with_keywords']
    without_keywords = [item for item in dataset if item.get('translation_mode') == 'without_keywords']
    
    print(f"Found {len(with_keywords)} translations with keywords")
    print(f"Found {len(without_keywords)} translations without keywords")
    
    # Create a mapping by file_path to match translations
    with_keywords_dict = {item['file_path']: item for item in with_keywords}
    without_keywords_dict = {item['file_path']: item for item in without_keywords}
    
    # Find common file paths
    common_paths = set(with_keywords_dict.keys()) & set(without_keywords_dict.keys())
    print(f"Found {len(common_paths)} files with both translation modes")
    
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    # Create comparison rows
    comparison_data = []
    for file_path in sorted(common_paths):
        with_kw = with_keywords_dict[file_path]
        without_kw = without_keywords_dict[file_path]
        
        # Determine if we should shuffle the order for this row
        is_shuffled = shuffle_order and random.choice([True, False])
        
        if is_shuffled:
            # Shuffle: put without_keywords first, with_keywords second
            translation_a = without_kw['translated_text']
            translation_b = with_kw['translated_text']
            mode_a = 'without_keywords'
            mode_b = 'with_keywords'
        else:
            # Normal order: with_keywords first, without_keywords second
            translation_a = with_kw['translated_text']
            translation_b = without_kw['translated_text']
            mode_a = 'with_keywords'
            mode_b = 'without_keywords'
        
        # Extract additional data from with_keywords (since it has the concept extraction data)
        extracted_concepts = with_kw.get('extracted_concepts', [])
        umls_lookups = with_kw.get('umls_lookups', [])
        concepts_requiring_umls = with_kw.get('concepts_requiring_umls', [])
        snomed_translations = with_kw.get('snomed_translations', [])
        timing = with_kw.get('timing', {})
        translation_metadata = with_kw.get('translation_metadata', {})
        prompt_details = with_kw.get('prompt_details', {})
        
        # Extract evaluation data
        with_kw_evaluation = with_kw.get('evaluation', {})
        without_kw_evaluation = without_kw.get('evaluation', {})
        # Support both COMET (comet_score) and LLM (llm_score) evaluations
        # Use explicit None check to handle score of 0 correctly
        with_kw_comet_score = with_kw_evaluation.get('comet_score') if 'comet_score' in with_kw_evaluation else (with_kw_evaluation.get('llm_score') if 'llm_score' in with_kw_evaluation else None)
        without_kw_comet_score = without_kw_evaluation.get('comet_score') if 'comet_score' in without_kw_evaluation else (without_kw_evaluation.get('llm_score') if 'llm_score' in without_kw_evaluation else None)
        
        # Format extracted concepts as readable text
        concepts_text = ""
        if extracted_concepts:
            concepts_list = []
            for concept in extracted_concepts:
                if isinstance(concept, dict):
                    # Extract term and category
                    term = concept.get('term', '') or ''
                    category = concept.get('category', '') or ''
                    confidence = concept.get('confidence_score', 0)
                    if term:
                        concept_info = f"{term} ({category}, conf: {confidence:.2f})"
                        concepts_list.append(concept_info)
                else:
                    concepts_list.append(str(concept) if concept is not None else '')
            concepts_text = "; ".join(concepts_list)
        
        # Format UMLS lookups as readable text
        umls_text = ""
        if umls_lookups:
            umls_list = []
            for lookup in umls_lookups:
                if isinstance(lookup, dict):
                    term = lookup.get('term', '') or ''
                    umls_cui = lookup.get('umls_cui', '') or ''
                    normalized_text = lookup.get('normalized_text', '') or ''
                    success = lookup.get('success', False)
                    if term:
                        status = "✓" if success else "✗"
                        umls_info = f"{term} → {normalized_text} (CUI: {umls_cui}) {status}"
                        umls_list.append(umls_info)
                else:
                    umls_list.append(str(lookup) if lookup is not None else '')
            umls_text = "; ".join(umls_list)
        
        # Format concepts requiring UMLS as readable text
        concepts_requiring_umls_text = ""
        if concepts_requiring_umls:
            concepts_umls_list = []
            for concept in concepts_requiring_umls:
                if isinstance(concept, dict):
                    term = concept.get('term', '') or ''
                    category = concept.get('category', '') or ''
                    umls_cui = concept.get('umls_cui', '') or ''
                    snomed_code = concept.get('snomed_code', '') or ''
                    success = concept.get('success', False)
                    confidence = concept.get('confidence_score', 0)
                    if term:
                        status = "✓" if success else "✗"
                        concept_info = f"{term} ({category}, conf: {confidence:.2f}) → CUI: {umls_cui} → SNOMED: {snomed_code} {status}"
                        concepts_umls_list.append(concept_info)
                else:
                    concepts_umls_list.append(str(concept) if concept is not None else '')
            concepts_requiring_umls_text = "; ".join(concepts_umls_list)
        
        # Format SNOMED translations as readable text
        snomed_text = ""
        if snomed_translations:
            snomed_list = []
            for translation in snomed_translations:
                if isinstance(translation, dict):
                    term = translation.get('term', '') or ''
                    translation_result = translation.get('translation', '') or ''
                    success = translation.get('success', False)
                    if term:
                        if success and translation_result:
                            snomed_info = f"{term} → {translation_result} ✓"
                        else:
                            snomed_info = f"{term} → [no translation] ✗"
                        snomed_list.append(snomed_info)
                else:
                    snomed_list.append(str(translation) if translation is not None else '')
            snomed_text = "; ".join(snomed_list)
        
        # Format prompt details
        prompt_text = ""
        if prompt_details:
            if isinstance(prompt_details, dict):
                system_prompt = prompt_details.get('system_prompt', '') or ''
                user_prompt = prompt_details.get('user_prompt', '') or ''
                prompt_text = system_prompt + "\n\n" + user_prompt
            else:
                prompt_text = str(prompt_details)
        
        row = {
            'file_path': file_path,
            'patient_id': with_kw['patient_id'],
            'study_id': with_kw['study_id'],
            'partition': with_kw['partition'],
            'model_used': with_kw['model_used'],
            'source_lang': with_kw.get('source_lang', ''),
            'target_lang': with_kw.get('target_lang', ''),
            'original_text': with_kw['original_text'],
            'translation_a': translation_a,
            'translation_b': translation_b,
            'translation_a_mode': mode_a,
            'translation_b_mode': mode_b,
            'is_shuffled': is_shuffled,
            # Concept extraction data
            'extracted_concepts': concepts_text,
            'num_extracted_concepts': with_kw.get('num_extracted_concepts', 0),
            'umls_lookups': umls_text,
            'concepts_requiring_umls': concepts_requiring_umls_text,
            'num_umls_lookups': with_kw.get('num_umls_lookups', 0),
            'snomed_translations': snomed_text,
            'num_snomed_translations': with_kw.get('num_snomed_translations', 0),
            # Timing data
            'extraction_time': timing.get('concept_extraction', 0),
            'translation_time': timing.get('translation', 0),
            'total_time': timing.get('total', 0),
            # Translation metadata
            'translation_metadata': str(translation_metadata),
            'prompt_details': prompt_text,
            # Evaluation data (supports both COMET and LLM scores)
            'with_keywords_comet_score': with_kw_comet_score,
            'without_keywords_comet_score': without_kw_comet_score,
            'comet_score_difference': (with_kw_comet_score - without_kw_comet_score) if (with_kw_comet_score is not None and without_kw_comet_score is not None) else None,
            'evaluation_model': with_kw_evaluation.get('model_name', '') or without_kw_evaluation.get('model_name', ''),
            'evaluation_timestamp': with_kw_evaluation.get('evaluation_timestamp', '') or without_kw_evaluation.get('evaluation_timestamp', ''),
            'evaluation_type': 'COMET' if ('comet_score' in with_kw_evaluation or 'comet_score' in without_kw_evaluation) else 'LLM',
            # Annotation columns
            'comparison_score': '',  # Numerical comparison: -2 to +2 scale
            'notes': ''  # For additional comments
        }
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


def create_excel_comparison(
    dataset_path: Optional[str] = None,
    output_path: str = "data/comparisons",
    filename: Optional[str] = None,
    shuffle_order: bool = True,
    random_seed: Optional[int] = None
) -> str:
    """
    Create an Excel file for comparing translation modes with evaluation scores.
    
    Args:
        dataset_path: Path to the translation dataset with evaluation (None to use latest)
        output_path: Directory to save the Excel file
        filename: Custom filename (None to auto-generate)
        shuffle_order: Whether to randomly shuffle the order of translation modes
        random_seed: Random seed for reproducible shuffling
        
    Returns:
        Path to the created Excel file
    """
    # Load dataset and metadata
    dataset, metadata = load_translation_dataset(dataset_path)
    
    # Create comparison DataFrame
    df = create_comparison_dataframe(dataset, shuffle_order=shuffle_order, random_seed=random_seed)
    
    if df.empty:
        raise EvaluationError("No matching translations found between the two modes")
    
    print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Get language info from the DataFrame if available, otherwise use metadata
        if len(df) > 0:
            source_lang = df['source_lang'].iloc[0] if 'source_lang' in df.columns and df['source_lang'].iloc[0] else metadata.get('source_lang', 'unknown')
            target_lang = df['target_lang'].iloc[0] if 'target_lang' in df.columns and df['target_lang'].iloc[0] else metadata.get('target_lang', 'unknown')
        else:
            source_lang = metadata.get('source_lang', 'unknown')
            target_lang = metadata.get('target_lang', 'unknown')
        filename = f"translation_comparison_{source_lang}_{target_lang}_{timestamp}.xlsx"
    
    excel_path = os.path.join(output_path, filename)
    
    # Create Excel file with multiple sheets
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

            # Main comparison sheet
            df.to_excel(writer, sheet_name='Translation Comparison', index=False)
            
            # Summary sheet
            shuffled_count = df['is_shuffled'].sum() if 'is_shuffled' in df.columns else 0
            summary_data = {
                'Metric': [
                    'Total Comparisons',
                    'Source Language',
                    'Target Language',
                    'Model Used',
                    'Translation Mode',
                    'Shuffled Rows',
                    'Shuffle Percentage',
                    'Random Seed Used',
                    'Dataset Timestamp',
                    'Generated Timestamp',
                    'Evaluation Model',
                    'Evaluation Timestamp',
                    'Evaluation Type',
                    'System Score',
                    'Mean Score'
                ],
                'Value': [
                    len(df),
                    metadata.get('source_lang', 'Unknown'),
                    metadata.get('target_lang', 'Unknown'),
                    df['model_used'].iloc[0] if len(df) > 0 else 'Unknown',
                    metadata.get('translation_mode', 'Unknown'),
                    shuffled_count,
                    f"{(shuffled_count / len(df) * 100):.1f}%" if len(df) > 0 else "0%",
                    random_seed if random_seed is not None else 'None',
                    metadata.get('timestamp', 'Unknown'),
                    datetime.now().isoformat(),
                    metadata.get('evaluation_model', 'Unknown'),
                    metadata.get('evaluation_timestamp', 'Unknown'),
                    df['evaluation_type'].iloc[0] if 'evaluation_type' in df.columns and len(df) > 0 else ('COMET' if any(item.get('evaluation', {}).get('comet_score') is not None for item in dataset) else 'LLM'),
                    metadata.get('system_score', 'N/A'),
                    metadata.get('mean_score', 'N/A')
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Statistics sheet
            # Calculate evaluation statistics
            with_kw_scores = df['with_keywords_comet_score'].dropna()
            without_kw_scores = df['without_keywords_comet_score'].dropna()
            score_differences = df['comet_score_difference'].dropna()
            
            stats_data = {
                'Metric': [
                    'Average Concepts Extracted',
                    'Average UMLS Lookups Required',
                    'Average SNOMED Translations',
                    'UMLS Lookup Success Rate',
                    'Average Extraction Time (s)',
                    'Average Translation Time (s)',
                    'Average Total Time (s)',
                    'Files with Keywords',
                    'Files without Keywords',
                    'Total UMLS Lookups',
                    'Total Concepts Requiring UMLS',
                    'Average Score (with_keywords)',
                    'Average Score (without_keywords)',
                    'Average Score Difference',
                    'Score Std Dev (with_keywords)',
                    'Score Std Dev (without_keywords)',
                    'Min Score (with_keywords)',
                    'Min Score (without_keywords)',
                    'Max Score (with_keywords)',
                    'Max Score (without_keywords)',
                    'Evaluations with Keywords',
                    'Evaluations without Keywords'
                ],
                'Value': [
                    df['num_extracted_concepts'].mean(),
                    df['num_umls_lookups'].mean(),
                    df['num_snomed_translations'].mean(),
                    f"{(df['num_umls_lookups'].sum() / max(df['num_extracted_concepts'].sum(), 1) * 100):.1f}%" if df['num_extracted_concepts'].sum() > 0 else "0%",
                    df['extraction_time'].mean(),
                    df['translation_time'].mean(),
                    df['total_time'].mean(),
                    len([x for x in dataset if x.get('translation_mode') == 'with_keywords']),
                    len([x for x in dataset if x.get('translation_mode') == 'without_keywords']),
                    df['num_umls_lookups'].sum(),
                    df['num_umls_lookups'].sum(),
                    with_kw_scores.mean() if len(with_kw_scores) > 0 else 'N/A',
                    without_kw_scores.mean() if len(without_kw_scores) > 0 else 'N/A',
                    score_differences.mean() if len(score_differences) > 0 else 'N/A',
                    with_kw_scores.std() if len(with_kw_scores) > 0 else 'N/A',
                    without_kw_scores.std() if len(without_kw_scores) > 0 else 'N/A',
                    with_kw_scores.min() if len(with_kw_scores) > 0 else 'N/A',
                    without_kw_scores.min() if len(without_kw_scores) > 0 else 'N/A',
                    with_kw_scores.max() if len(with_kw_scores) > 0 else 'N/A',
                    without_kw_scores.max() if len(without_kw_scores) > 0 else 'N/A',
                    len(with_kw_scores),
                    len(without_kw_scores)
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        print(f"Excel comparison file created: {excel_path}")
        print(f"Total comparisons: {len(df)}")
        
        return excel_path
        
    except Exception as e:
        raise EvaluationError(f"Failed to create Excel file: {str(e)}")


def main():
    """Command line interface for creating Excel comparison files."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Excel file for comparing translation modes with evaluation scores.')
    parser.add_argument('--dataset-path', 
                       help='Path to the translation dataset with evaluation (default: latest)')
    parser.add_argument('--output-dir', default='data/comparisons', 
                       help='Directory to save the Excel file')
    parser.add_argument('--filename', 
                       help='Custom filename for the Excel file')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Disable shuffling of translation order')
    parser.add_argument('--random-seed', type=int,
                       help='Random seed for reproducible shuffling')
    
    args = parser.parse_args()
    
    try:
        excel_path = create_excel_comparison(
            dataset_path=args.dataset_path,
            output_path=args.output_dir,
            filename=args.filename,
            shuffle_order=not args.no_shuffle,
            random_seed=args.random_seed
        )
        print(f"\nSuccess! Excel file created at: {excel_path}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
