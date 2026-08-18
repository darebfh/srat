"""
Script to export all extracted concepts from 30 reports to Excel for manual precision calculation.
One concept per line with report context.
"""

import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any


def get_unique_reports(translations_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load translations and get all unique reports by study_id.
    
    Returns:
        Dictionary mapping study_id to translation (preferring entries with concepts)
    """
    with open(translations_path, 'r', encoding='utf-8') as f:
        all_translations = json.load(f)
    
    # Get unique reports by study_id
    # Use a dictionary to keep track of seen study_ids
    # Prefer entries that have extracted_concepts
    seen_study_ids = {}
    
    for translation in all_translations:
        study_id = translation.get('study_id', 'Unknown')
        
        # If we haven't seen this study_id yet, add it
        if study_id not in seen_study_ids:
            seen_study_ids[study_id] = translation
        else:
            # If we've seen it, prefer the one with extracted_concepts
            existing = seen_study_ids[study_id]
            existing_has_concepts = 'extracted_concepts' in existing and existing.get('extracted_concepts') and len(existing.get('extracted_concepts', [])) > 0
            current_has_concepts = 'extracted_concepts' in translation and translation.get('extracted_concepts') and len(translation.get('extracted_concepts', [])) > 0
            
            # Replace if current has concepts and existing doesn't
            if current_has_concepts and not existing_has_concepts:
                seen_study_ids[study_id] = translation
    
    return seen_study_ids


def load_translations(translations_path: Path, limit: int = 30) -> List[Dict[str, Any]]:
    """
    Load translations from JSON file and return N unique reports.
    
    Args:
        translations_path: Path to translations JSON file
        limit: Number of unique reports to return (default 30)
    
    Returns:
        List of N unique translations (one per study_id)
    """
    seen_study_ids = get_unique_reports(translations_path)
    
    # Convert to list and take first N unique reports
    unique_translations = list(seen_study_ids.values())[:limit]
    
    print(f"Found {len(seen_study_ids)} unique reports, selecting first {limit}")
    return unique_translations


def extract_all_concepts(translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all concepts from all translations, one concept per line.
    
    Returns:
        List of dictionaries, each representing one concept with report context
    """
    all_concepts = []
    
    for translation in translations:
        # Get report identification
        study_id = translation.get('study_id', 'Unknown')
        file_path = translation.get('file_path', 'Unknown')
        patient_id = translation.get('patient_id', 'Unknown')
        partition = translation.get('partition', 'Unknown')
        
        # Get concepts
        concepts = translation.get('extracted_concepts', [])
        
        if not concepts:
            continue
        
        # Create one row per concept
        for concept in concepts:
            concept_row = {
                'study_id': study_id,
                'patient_id': patient_id,
                'file_path': file_path,
                'partition': partition,
                'term': concept.get('term', ''),
                'normalized_text': concept.get('normalized_text', ''),
                'category': concept.get('category', ''),
                'confidence_score': concept.get('confidence_score', 0.0),
                'codes': str(concept.get('codes', {})) if concept.get('codes') else '',
                'snomed_code': concept.get('codes', {}).get('SNOMEDCT', '') if concept.get('codes') else '',
                'umls_cui': concept.get('codes', {}).get('UMLS', '') if concept.get('codes') else '',
                # Column for manual annotation
                'is_correct': '',  # Empty for manual annotation (True/False)
                'notes': ''  # For additional notes
            }
            all_concepts.append(concept_row)
    
    return all_concepts


def generate_all_reports_statistics(all_unique_reports: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics for all unique reports.
    
    Args:
        all_unique_reports: Dictionary of all unique reports by study_id
        
    Returns:
        Dictionary with statistics
    """
    all_concepts_all_reports = []
    reports_with_concepts = 0
    reports_without_concepts = 0
    
    for study_id, translation in all_unique_reports.items():
        concepts = translation.get('extracted_concepts', [])
        if concepts and len(concepts) > 0:
            reports_with_concepts += 1
            all_concepts_all_reports.extend(concepts)
        else:
            reports_without_concepts += 1
    
    if not all_concepts_all_reports:
        return {
            'total_reports': len(all_unique_reports),
            'reports_with_concepts': 0,
            'reports_without_concepts': len(all_unique_reports),
            'total_concepts': 0,
            'unique_terms': 0,
            'unique_normalized_terms': 0,
            'avg_concepts_per_report': 0,
            'avg_confidence': 0,
            'concepts_with_snomed': 0,
            'concepts_with_umls': 0,
            'categories': []
        }
    
    # Calculate statistics
    unique_terms = set()
    unique_normalized_terms = set()
    confidence_scores = []
    concepts_with_snomed = 0
    concepts_with_umls = 0
    categories = {}
    
    for concept in all_concepts_all_reports:
        term = concept.get('term', '')
        normalized = concept.get('normalized_text', '')
        confidence = concept.get('confidence_score', 0.0)
        codes = concept.get('codes', {})
        category = concept.get('category', 'Unknown')
        
        if term:
            unique_terms.add(term)
        if normalized:
            unique_normalized_terms.add(normalized)
        
        if isinstance(confidence, (int, float)):
            confidence_scores.append(confidence)
        
        if codes:
            if codes.get('SNOMEDCT'):
                concepts_with_snomed += 1
            if codes.get('UMLS'):
                concepts_with_umls += 1
        
        categories[category] = categories.get(category, 0) + 1
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    avg_concepts_per_report = len(all_concepts_all_reports) / reports_with_concepts if reports_with_concepts > 0 else 0
    
    return {
        'total_reports': len(all_unique_reports),
        'reports_with_concepts': reports_with_concepts,
        'reports_without_concepts': reports_without_concepts,
        'total_concepts': len(all_concepts_all_reports),
        'unique_terms': len(unique_terms),
        'unique_normalized_terms': len(unique_normalized_terms),
        'avg_concepts_per_report': avg_concepts_per_report,
        'avg_confidence': avg_confidence,
        'concepts_with_snomed': concepts_with_snomed,
        'concepts_with_umls': concepts_with_umls,
        'categories': categories
    }


def create_excel_file(concepts: List[Dict[str, Any]], output_path: Path, all_reports_stats: Dict[str, Any] = None) -> None:
    """
    Create Excel file with all concepts, one per line.
    
    Args:
        concepts: List of concept dictionaries
        output_path: Path to save the Excel file
    """
    if not concepts:
        print("No concepts found to export.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(concepts)
    
    # Reorder columns for better readability
    column_order = [
        'study_id',
        'patient_id',
        'file_path',
        'partition',
        'term',
        'normalized_text',
        'category',
        'confidence_score',
        'snomed_code',
        'umls_cui',
        'codes',
        'is_correct',
        'notes'
    ]
    
    # Select only columns that exist
    display_columns = [col for col in column_order if col in df.columns]
    df = df[display_columns]
    
    # Keep confidence score as numeric for calculations, but format for display
    # We'll keep it numeric in the main sheet for easier filtering/sorting
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create Excel file
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main concepts sheet
        df.to_excel(writer, sheet_name='Concepts', index=False)
        
        # Summary sheet
        summary_data = {
            'Metric': [
                'Total Concepts',
                'Unique Terms',
                'Unique Normalized Terms',
                'Total Reports',
                'Average Concepts per Report',
                'Categories',
                'Average Confidence Score',
                'Concepts with SNOMED Code',
                'Concepts with UMLS CUI',
                'Generated Timestamp'
            ],
            'Value': [
                len(df),
                df['term'].nunique() if 'term' in df.columns else 0,
                df['normalized_text'].nunique() if 'normalized_text' in df.columns else 0,
                df['study_id'].nunique() if 'study_id' in df.columns else 0,
                len(df) / df['study_id'].nunique() if 'study_id' in df.columns and df['study_id'].nunique() > 0 else 0,
                ', '.join(sorted(df['category'].unique().tolist())) if 'category' in df.columns else '',
                pd.to_numeric(df['confidence_score'], errors='coerce').mean() if 'confidence_score' in df.columns else 0,
                (df['snomed_code'] != '').sum() if 'snomed_code' in df.columns else 0,
                (df['umls_cui'] != '').sum() if 'umls_cui' in df.columns else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary (30 Reports)', index=False)
        
        # Add statistics for all 500 reports if provided
        if all_reports_stats:
            all_reports_summary_data = {
                'Metric': [
                    'Total Reports (All)',
                    'Reports with Concepts',
                    'Reports without Concepts',
                    'Total Concepts (All)',
                    'Unique Terms (All)',
                    'Unique Normalized Terms (All)',
                    'Average Concepts per Report (All)',
                    'Average Confidence Score (All)',
                    'Concepts with SNOMED Code (All)',
                    'Concepts with UMLS CUI (All)',
                    'Percentage of Reports with Concepts',
                    'Generated Timestamp'
                ],
                'Value': [
                    all_reports_stats['total_reports'],
                    all_reports_stats['reports_with_concepts'],
                    all_reports_stats['reports_without_concepts'],
                    all_reports_stats['total_concepts'],
                    all_reports_stats['unique_terms'],
                    all_reports_stats['unique_normalized_terms'],
                    f"{all_reports_stats['avg_concepts_per_report']:.2f}",
                    f"{all_reports_stats['avg_confidence']:.4f}",
                    all_reports_stats['concepts_with_snomed'],
                    all_reports_stats['concepts_with_umls'],
                    f"{(all_reports_stats['reports_with_concepts'] / all_reports_stats['total_reports'] * 100):.2f}%" if all_reports_stats['total_reports'] > 0 else "0%",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            }
            all_reports_summary_df = pd.DataFrame(all_reports_summary_data)
            all_reports_summary_df.to_excel(writer, sheet_name='Summary (All Reports)', index=False)
            
            # Category breakdown for all reports
            if all_reports_stats['categories']:
                category_data = []
                for category, count in sorted(all_reports_stats['categories'].items(), key=lambda x: x[1], reverse=True):
                    category_data.append({
                        'Category': category,
                        'Count': count,
                        'Percentage': f"{(count / all_reports_stats['total_concepts'] * 100):.2f}%"
                    })
                category_all_df = pd.DataFrame(category_data)
                category_all_df.to_excel(writer, sheet_name='Category Breakdown (All)', index=False)
        
        # Category breakdown sheet (for 30 reports)
        if 'category' in df.columns:
            category_stats = df.groupby('category').agg({
                'term': 'count',
                'confidence_score': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                'snomed_code': lambda x: (x != '').sum(),
                'umls_cui': lambda x: (x != '').sum()
            }).reset_index()
            category_stats.columns = ['Category', 'Count', 'Avg Confidence', 'With SNOMED', 'With UMLS']
            category_stats = category_stats.sort_values('Count', ascending=False)
            # Format confidence for display
            if 'Avg Confidence' in category_stats.columns:
                category_stats['Avg Confidence'] = category_stats['Avg Confidence'].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
            category_stats.to_excel(writer, sheet_name='Category Breakdown (30 Reports)', index=False)
    
    print(f"Excel file created successfully: {output_path}")
    print(f"Total concepts exported: {len(df)}")
    print(f"Total reports: {df['study_id'].nunique() if 'study_id' in df.columns else 0}")


def main():
    """Main function to export concepts to Excel."""
    # Hardcoded path
    translations_path = Path("data/results/00_gpt-oss:120b_nodict_baseprompt/20250928_092100/translations.json")
    
    # Output path
    output_dir = Path("data/exports")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"concepts_for_precision_{timestamp}.xlsx"
    
    print(f"Loading translations from: {translations_path}")
    
    if not translations_path.exists():
        print(f"Error: Translations file not found at: {translations_path}")
        return
    
    # Get all unique reports for statistics
    print("Loading all unique reports for statistics...")
    all_unique_reports = get_unique_reports(translations_path)
    print(f"Found {len(all_unique_reports)} unique reports")
    
    # Generate statistics for all reports
    print("Generating statistics for all reports...")
    all_reports_stats = generate_all_reports_statistics(all_unique_reports)
    print(f"Statistics: {all_reports_stats['total_concepts']} concepts from {all_reports_stats['reports_with_concepts']} reports")
    
    # Load first 30 translations for manual annotation
    translations = load_translations(translations_path, limit=30)
    print(f"Loaded {len(translations)} translations for manual annotation")
    
    # Extract all concepts from the 30 reports
    print("Extracting concepts from 30 reports...")
    all_concepts = extract_all_concepts(translations)
    print(f"Found {len(all_concepts)} total concepts from 30 reports")
    
    # Create Excel file with both 30 reports data and all reports statistics
    print("Creating Excel file...")
    create_excel_file(all_concepts, output_path, all_reports_stats)
    
    print("\nDone! The Excel file contains:")
    print(f"  - Concepts from 30 reports for manual annotation (sheet: 'Concepts')")
    print(f"  - Statistics for 30 reports (sheet: 'Summary (30 Reports)')")
    print(f"  - Statistics for all {len(all_unique_reports)} reports (sheet: 'Summary (All Reports)')")
    print(f"  - Category breakdowns for both datasets")
    print("\nYou can now manually annotate the 'is_correct' column in the Excel file.")
    print("Use True/False values to mark whether each concept extraction is correct.")


if __name__ == "__main__":
    main()

