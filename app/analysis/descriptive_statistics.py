"""
Descriptive Statistics Module for Medical Report Translation Dataset

This module provides comprehensive descriptive statistics for the subset of 500 reports
used in the translation experiments. It analyzes various aspects including:
- Basic dataset statistics (counts, sizes, distributions)
- Text analysis (length, word counts, readability metrics)
- Medical-specific analysis (partition distribution, patient analysis)
- Export capabilities for results
"""

import os
import json
import re
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import statistics
import math

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualization features will be limited.")

try:
    from readability import Readability
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    print("Warning: py-readability-metrics not available. Using simplified readability calculation.")

try:
    from datasets import load_from_disk
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Cannot load Arrow datasets.")

from exceptions import EvaluationError


class DescriptiveStatistics:
    """Comprehensive descriptive statistics for medical report datasets."""
    
    def __init__(self, dataset_path: str = "data/processed/raw_dataset"):
        """
        Initialize the descriptive statistics analyzer.
        
        Args:
            dataset_path: Path to the dataset (Arrow format)
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.stats = {}
        
    def load_dataset(self, num_reports: Optional[int] = None, random_seed: Optional[int] = None) -> None:
        """
        Load the dataset from disk.
        
        Args:
            num_reports: Number of reports to select (None for all reports)
            random_seed: Random seed for reproducible selection
        """
        if not DATASETS_AVAILABLE:
            raise EvaluationError("datasets library not available. Cannot load Arrow datasets.")
        
        if not os.path.exists(self.dataset_path):
            raise EvaluationError(f"Dataset not found: {self.dataset_path}")
        
        print(f"Loading dataset from: {self.dataset_path}")
        full_dataset = load_from_disk(self.dataset_path)
        
        # Select subset if requested
        if num_reports is not None and num_reports < len(full_dataset):
            import random
            if random_seed is not None:
                random.seed(random_seed)
                print(f"Using random seed {random_seed} for subset selection")
            
            indices = random.sample(range(len(full_dataset)), num_reports)
            self.dataset = full_dataset.select(indices)
            print(f"Selected {len(self.dataset)} reports from {len(full_dataset)} total reports")
        else:
            self.dataset = full_dataset
            print(f"Loaded {len(self.dataset)} reports")
    
    def get_basic_statistics(self, num_reports: Optional[int] = None, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate basic dataset statistics.
        
        Args:
            num_reports: Number of reports to analyze (None for all reports)
            random_seed: Random seed for reproducible subset selection
        
        Returns:
            Dictionary containing basic statistics
        """
        if self.dataset is None:
            self.load_dataset(num_reports, random_seed)
        
        # Basic counts
        total_reports = len(self.dataset)
        unique_patients = len(set(self.dataset['patient_id']))
        unique_studies = len(set(self.dataset['study_id']))
        
        # File size statistics
        file_sizes = self.dataset['file_size']
        total_size_bytes = sum(file_sizes)
        total_size_mb = total_size_bytes / (1024 * 1024)
        avg_size_bytes = statistics.mean(file_sizes)
        median_size_bytes = statistics.median(file_sizes)
        std_size_bytes = statistics.stdev(file_sizes) if len(file_sizes) > 1 else 0
        
        # Text length statistics
        text_lengths = [len(text) for text in self.dataset['text']]
        avg_text_length = statistics.mean(text_lengths)
        median_text_length = statistics.median(text_lengths)
        std_text_length = statistics.stdev(text_lengths) if len(text_lengths) > 1 else 0
        min_text_length = min(text_lengths)
        max_text_length = max(text_lengths)
        
        # Word count statistics
        word_counts = []
        for text in self.dataset['text']:
            words = len(text.split())
            word_counts.append(words)
        
        avg_word_count = statistics.mean(word_counts)
        median_word_count = statistics.median(word_counts)
        std_word_count = statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        min_word_count = min(word_counts)
        max_word_count = max(word_counts)
        
        return {
            'total_reports': total_reports,
            'unique_patients': unique_patients,
            'unique_studies': unique_studies,
            'file_size': {
                'total_bytes': total_size_bytes,
                'total_mb': round(total_size_mb, 2),
                'avg_bytes': round(avg_size_bytes, 2),
                'median_bytes': round(median_size_bytes, 2),
                'std_bytes': round(std_size_bytes, 2),
                'min_bytes': min(file_sizes),
                'max_bytes': max(file_sizes)
            },
            'text_length': {
                'avg_characters': round(avg_text_length, 2),
                'median_characters': round(median_text_length, 2),
                'std_characters': round(std_text_length, 2),
                'min_characters': min_text_length,
                'max_characters': max_text_length
            },
            'word_count': {
                'avg_words': round(avg_word_count, 2),
                'median_words': round(median_word_count, 2),
                'std_words': round(std_word_count, 2),
                'min_words': min_word_count,
                'max_words': max_word_count
            }
        }
    
    def get_partition_analysis(self) -> Dict[str, Any]:
        """
        Analyze distribution by partition.
        
        Returns:
            Dictionary containing partition analysis
        """
        if self.dataset is None:
            self.load_dataset()
        
        partition_counts = Counter(self.dataset['partition'])
        total_reports = len(self.dataset)
        
        partition_stats = {}
        for partition, count in partition_counts.items():
            percentage = (count / total_reports) * 100
            partition_stats[partition] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        return {
            'total_partitions': len(partition_counts),
            'partition_distribution': partition_stats,
            'most_common_partition': partition_counts.most_common(1)[0] if partition_counts else None,
            'least_common_partition': partition_counts.most_common()[-1] if partition_counts else None
        }
    
    def get_patient_analysis(self) -> Dict[str, Any]:
        """
        Analyze patient distribution and statistics.
        
        Returns:
            Dictionary containing patient analysis
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Count reports per patient
        patient_counts = Counter(self.dataset['patient_id'])
        
        # Calculate statistics
        reports_per_patient = list(patient_counts.values())
        avg_reports_per_patient = statistics.mean(reports_per_patient)
        median_reports_per_patient = statistics.median(reports_per_patient)
        std_reports_per_patient = statistics.stdev(reports_per_patient) if len(reports_per_patient) > 1 else 0
        max_reports_per_patient = max(reports_per_patient)
        min_reports_per_patient = min(reports_per_patient)
        
        # Find patients with most/least reports
        most_reports_patient = patient_counts.most_common(1)[0] if patient_counts else None
        least_reports_patient = patient_counts.most_common()[-1] if patient_counts else None
        
        # Patients with multiple reports
        patients_with_multiple_reports = sum(1 for count in reports_per_patient if count > 1)
        percentage_multiple_reports = (patients_with_multiple_reports / len(patient_counts)) * 100
        
        return {
            'total_patients': len(patient_counts),
            'reports_per_patient': {
                'avg': round(avg_reports_per_patient, 2),
                'median': round(median_reports_per_patient, 2),
                'std': round(std_reports_per_patient, 2),
                'min': min_reports_per_patient,
                'max': max_reports_per_patient
            },
            'most_reports_patient': most_reports_patient,
            'least_reports_patient': least_reports_patient,
            'patients_with_multiple_reports': {
                'count': patients_with_multiple_reports,
                'percentage': round(percentage_multiple_reports, 2)
            }
        }
    
    def get_text_analysis(self) -> Dict[str, Any]:
        """
        Perform detailed text analysis.
        
        Returns:
            Dictionary containing text analysis
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Character-level analysis
        char_counts = [len(text) for text in self.dataset['text']]
        
        # Word-level analysis
        word_counts = []
        sentence_counts = []
        paragraph_counts = []
        
        for text in self.dataset['text']:
            # Word count
            words = text.split()
            word_counts.append(len(words))
            
            # Sentence count (simple heuristic)
            sentences = re.split(r'[.!?]+', text)
            sentence_counts.append(len([s for s in sentences if s.strip()]))
            
            # Paragraph count
            paragraphs = text.split('\n\n')
            paragraph_counts.append(len([p for p in paragraphs if p.strip()]))
        
        # Calculate readability metrics using py-readability-metrics
        readability_scores = []
        flesch_kincaid_scores = []
        ari_scores = []
        smog_scores = []
        
        for text in self.dataset['text']:
            try:
                if READABILITY_AVAILABLE and len(text.strip()) > 0:
                    r = Readability(text)
                    
                    # Flesch Reading Ease
                    try:
                        flesch_score = r.flesch()
                        readability_scores.append(flesch_score.score)
                    except:
                        readability_scores.append(0)
                    
                    # Flesch-Kincaid Grade Level
                    try:
                        fk_score = r.flesch_kincaid()
                        flesch_kincaid_scores.append(fk_score.score)
                    except:
                        flesch_kincaid_scores.append(0)
                    
                    # Automated Readability Index
                    try:
                        ari_score = r.ari()
                        ari_scores.append(ari_score.score)
                    except:
                        ari_scores.append(0)
                    
                    # SMOG Index
                    try:
                        smog_score = r.smog()
                        smog_scores.append(smog_score.score)
                    except:
                        smog_scores.append(0)
                else:
                    # Fallback to simplified calculation
                    words = text.split()
                    sentences = re.split(r'[.!?]+', text)
                    sentences = [s for s in sentences if s.strip()]
                    
                    if len(words) > 0 and len(sentences) > 0:
                        syllables = sum(self._count_syllables(word) for word in words)
                        score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
                        readability_scores.append(score)
                        flesch_kincaid_scores.append(0)
                        ari_scores.append(0)
                        smog_scores.append(0)
                    else:
                        readability_scores.append(0)
                        flesch_kincaid_scores.append(0)
                        ari_scores.append(0)
                        smog_scores.append(0)
            except Exception as e:
                # Handle any errors in readability calculation
                readability_scores.append(0)
                flesch_kincaid_scores.append(0)
                ari_scores.append(0)
                smog_scores.append(0)
        
        return {
            'character_analysis': {
                'avg_characters': round(statistics.mean(char_counts), 2),
                'median_characters': round(statistics.median(char_counts), 2),
                'std_characters': round(statistics.stdev(char_counts), 2) if len(char_counts) > 1 else 0,
                'min_characters': min(char_counts),
                'max_characters': max(char_counts)
            },
            'word_analysis': {
                'avg_words': round(statistics.mean(word_counts), 2),
                'median_words': round(statistics.median(word_counts), 2),
                'std_words': round(statistics.stdev(word_counts), 2) if len(word_counts) > 1 else 0,
                'min_words': min(word_counts),
                'max_words': max(word_counts)
            },
            'sentence_analysis': {
                'avg_sentences': round(statistics.mean(sentence_counts), 2),
                'median_sentences': round(statistics.median(sentence_counts), 2),
                'std_sentences': round(statistics.stdev(sentence_counts), 2) if len(sentence_counts) > 1 else 0,
                'min_sentences': min(sentence_counts),
                'max_sentences': max(sentence_counts)
            },
            'paragraph_analysis': {
                'avg_paragraphs': round(statistics.mean(paragraph_counts), 2),
                'median_paragraphs': round(statistics.median(paragraph_counts), 2),
                'std_paragraphs': round(statistics.stdev(paragraph_counts), 2) if len(paragraph_counts) > 1 else 0,
                'min_paragraphs': min(paragraph_counts),
                'max_paragraphs': max(paragraph_counts)
            },
            'readability': {
                'flesch_reading_ease': {
                    'avg_score': round(statistics.mean(readability_scores), 2),
                    'median_score': round(statistics.median(readability_scores), 2),
                    'std_score': round(statistics.stdev(readability_scores), 2) if len(readability_scores) > 1 else 0,
                    'min_score': round(min(readability_scores), 2),
                    'max_score': round(max(readability_scores), 2)
                },
                'flesch_kincaid_grade_level': {
                    'avg_score': round(statistics.mean(flesch_kincaid_scores), 2),
                    'median_score': round(statistics.median(flesch_kincaid_scores), 2),
                    'std_score': round(statistics.stdev(flesch_kincaid_scores), 2) if len(flesch_kincaid_scores) > 1 else 0,
                    'min_score': round(min(flesch_kincaid_scores), 2),
                    'max_score': round(max(flesch_kincaid_scores), 2)
                },
                'automated_readability_index': {
                    'avg_score': round(statistics.mean(ari_scores), 2),
                    'median_score': round(statistics.median(ari_scores), 2),
                    'std_score': round(statistics.stdev(ari_scores), 2) if len(ari_scores) > 1 else 0,
                    'min_score': round(min(ari_scores), 2),
                    'max_score': round(max(ari_scores), 2)
                },
                'smog_index': {
                    'avg_score': round(statistics.mean(smog_scores), 2),
                    'median_score': round(statistics.median(smog_scores), 2),
                    'std_score': round(statistics.stdev(smog_scores), 2) if len(smog_scores) > 1 else 0,
                    'min_score': round(min(smog_scores), 2),
                    'max_score': round(max(smog_scores), 2)
                }
            }
        }
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified heuristic).
        
        Args:
            word: Word to count syllables for
            
        Returns:
            Number of syllables
        """
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def get_medical_terminology_analysis(self) -> Dict[str, Any]:
        """
        Analyze medical terminology and common medical terms.
        
        Returns:
            Dictionary containing medical terminology analysis
        """
        if self.dataset is None:
            self.load_dataset()
        
        # Common medical terms to look for
        medical_terms = {
            'anatomy': ['heart', 'lung', 'chest', 'abdomen', 'brain', 'spine', 'bone', 'muscle', 'artery', 'vein'],
            'procedures': ['surgery', 'biopsy', 'scan', 'x-ray', 'mri', 'ct', 'ultrasound', 'endoscopy'],
            'conditions': ['fracture', 'tumor', 'cancer', 'infection', 'inflammation', 'disease', 'syndrome'],
            'medications': ['mg', 'ml', 'dose', 'medication', 'drug', 'therapy', 'treatment'],
            'measurements': ['mm', 'cm', 'kg', 'mg', 'ml', 'degree', 'percent', '%']
        }
        
        term_counts = {}
        for category, terms in medical_terms.items():
            term_counts[category] = {}
            for term in terms:
                count = 0
                for text in self.dataset['text']:
                    count += text.lower().count(term.lower())
                term_counts[category][term] = count
        
        # Calculate total medical term occurrences
        total_medical_terms = sum(sum(category_counts.values()) for category_counts in term_counts.values())
        
        return {
            'total_medical_term_occurrences': total_medical_terms,
            'medical_term_categories': term_counts,
            'most_common_medical_terms': self._get_most_common_terms(term_counts)
        }
    
    def _get_most_common_terms(self, term_counts: Dict[str, Dict[str, int]]) -> List[Tuple[str, int]]:
        """
        Get the most common medical terms across all categories.
        
        Args:
            term_counts: Dictionary of term counts by category
            
        Returns:
            List of tuples (term, count) sorted by count
        """
        all_terms = []
        for category_counts in term_counts.values():
            all_terms.extend(category_counts.items())
        
        return sorted(all_terms, key=lambda x: x[1], reverse=True)[:20]
    
    def get_comprehensive_statistics(self, num_reports: Optional[int] = None, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics combining all analyses.
        
        Args:
            num_reports: Number of reports to analyze (None for all reports)
            random_seed: Random seed for reproducible subset selection
        
        Returns:
            Dictionary containing all statistics
        """
        print("Calculating comprehensive statistics...")
        
        stats = {
            'dataset_info': {
                'dataset_path': self.dataset_path,
                'num_reports_analyzed': num_reports,
                'random_seed_used': random_seed,
                'analysis_timestamp': self._get_timestamp()
            },
            'basic_statistics': self.get_basic_statistics(num_reports, random_seed),
            'partition_analysis': self.get_partition_analysis(),
            'patient_analysis': self.get_patient_analysis(),
            'text_analysis': self.get_text_analysis(),
            'medical_terminology_analysis': self.get_medical_terminology_analysis()
        }
        
        self.stats = stats
        return stats
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def print_summary(self) -> None:
        """Print a summary of the statistics."""
        if not self.stats:
            self.get_comprehensive_statistics()
        
        print("\n" + "="*60)
        print("DESCRIPTIVE STATISTICS SUMMARY")
        print("="*60)
        
        # Basic statistics
        basic = self.stats['basic_statistics']
        print(f"\nDataset Overview:")
        print(f"  Total Reports: {basic['total_reports']:,}")
        print(f"  Unique Patients: {basic['unique_patients']:,}")
        print(f"  Unique Studies: {basic['unique_studies']:,}")
        print(f"  Total Size: {basic['file_size']['total_mb']:.2f} MB")
        
        # Text statistics
        print(f"\nText Statistics:")
        print(f"  Average Length: {basic['text_length']['avg_characters']:.0f} characters")
        print(f"  Average Words: {basic['word_count']['avg_words']:.0f} words")
        print(f"  Median Words: {basic['word_count']['median_words']:.0f} words")
        
        # Partition analysis
        partition = self.stats['partition_analysis']
        print(f"\nPartition Distribution:")
        for part, data in partition['partition_distribution'].items():
            print(f"  {part}: {data['count']} reports ({data['percentage']:.1f}%)")
        
        # Patient analysis
        patient = self.stats['patient_analysis']
        print(f"\nPatient Statistics:")
        print(f"  Average Reports per Patient: {patient['reports_per_patient']['avg']:.2f}")
        print(f"  Patients with Multiple Reports: {patient['patients_with_multiple_reports']['count']} ({patient['patients_with_multiple_reports']['percentage']:.1f}%)")
        
        # Medical terminology
        medical = self.stats['medical_terminology_analysis']
        print(f"\nMedical Terminology:")
        print(f"  Total Medical Term Occurrences: {medical['total_medical_term_occurrences']:,}")
        print(f"  Top 5 Most Common Terms:")
        for i, (term, count) in enumerate(medical['most_common_medical_terms'][:5], 1):
            print(f"    {i}. {term}: {count} occurrences")
        
        print("="*60)
    
    def save_results(self, output_path: str, format: str = 'json') -> None:
        """
        Save results to file.
        
        Args:
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'excel')
        """
        if not self.stats:
            self.get_comprehensive_statistics()
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv' and PANDAS_AVAILABLE:
            # Create a flattened version for CSV
            flattened_data = self._flatten_stats_for_csv()
            df = pd.DataFrame(flattened_data)
            df.to_csv(output_path, index=False)
        
        elif format == 'excel' and PANDAS_AVAILABLE:
            # Create multiple sheets for Excel
            try:
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    # Summary sheet
                    summary_data = self._create_summary_dataframe()
                    summary_data.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Detailed statistics
                    detailed_data = self._create_detailed_dataframe()
                    detailed_data.to_excel(writer, sheet_name='Detailed_Stats', index=False)
                    
                    # Medical terminology
                    medical_data = self._create_medical_dataframe()
                    medical_data.to_excel(writer, sheet_name='Medical_Terms', index=False)
            except Exception as e:
                print(f"Warning: Excel export failed: {e}")
                print("Falling back to JSON format...")
                # Fallback to JSON
                json_output_path = output_path.replace('.xlsx', '.json').replace('.xls', '.json')
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, indent=2, ensure_ascii=False)
                print(f"Results saved to: {json_output_path}")
                return
        
        else:
            raise ValueError(f"Unsupported format: {format} or required libraries not available")
        
        print(f"Results saved to: {output_path}")
    
    def _flatten_stats_for_csv(self) -> List[Dict[str, Any]]:
        """Flatten statistics for CSV export."""
        # This is a simplified version - you might want to expand this
        return [{
            'metric': 'total_reports',
            'value': self.stats['basic_statistics']['total_reports'],
            'category': 'basic'
        }]
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame for Excel export."""
        basic = self.stats['basic_statistics']
        partition = self.stats['partition_analysis']
        patient = self.stats['patient_analysis']
        
        data = [
            ['Total Reports', basic['total_reports']],
            ['Unique Patients', basic['unique_patients']],
            ['Unique Studies', basic['unique_studies']],
            ['Total Size (MB)', basic['file_size']['total_mb']],
            ['Average Text Length', basic['text_length']['avg_characters']],
            ['Average Word Count', basic['word_count']['avg_words']],
            ['Total Partitions', partition['total_partitions']],
            ['Average Reports per Patient', patient['reports_per_patient']['avg']],
            ['Patients with Multiple Reports', patient['patients_with_multiple_reports']['count']]
        ]
        
        return pd.DataFrame(data, columns=['Metric', 'Value'])
    
    def _create_detailed_dataframe(self) -> pd.DataFrame:
        """Create detailed statistics DataFrame for Excel export."""
        # This would contain more detailed breakdowns
        return pd.DataFrame()
    
    def _create_medical_dataframe(self) -> pd.DataFrame:
        """Create medical terminology DataFrame for Excel export."""
        medical = self.stats['medical_terminology_analysis']
        
        data = []
        for category, terms in medical['medical_term_categories'].items():
            for term, count in terms.items():
                data.append([category, term, count])
        
        return pd.DataFrame(data, columns=['Category', 'Term', 'Count'])
    
    def create_visualizations(self, output_dir: str = "visualizations") -> None:
        """
        Create visualizations for the statistics.
        
        Args:
            output_dir: Directory to save visualization files
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib/seaborn not available. Cannot create visualizations.")
            return
        
        if not self.stats:
            self.get_comprehensive_statistics()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Partition Distribution
        self._create_partition_distribution_plot(output_dir)
        
        # 2. Text Length Distribution
        self._create_text_length_distribution_plot(output_dir)
        
        # 3. Word Count Distribution
        self._create_word_count_distribution_plot(output_dir)
        
        # 4. Medical Terminology Analysis
        self._create_medical_terminology_plot(output_dir)
        
        # 5. Patient Report Distribution
        self._create_patient_distribution_plot(output_dir)
        
        print(f"Visualizations saved to: {output_dir}")
    
    def _create_partition_distribution_plot(self, output_dir: str) -> None:
        """Create partition distribution pie chart."""
        partition_data = self.stats['partition_analysis']['partition_distribution']
        
        plt.figure(figsize=(10, 8))
        labels = list(partition_data.keys())
        sizes = [data['count'] for data in partition_data.values()]
        colors = sns.color_palette("husl", len(labels))
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Distribution of Reports by Partition', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'partition_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_text_length_distribution_plot(self, output_dir: str) -> None:
        """Create text length distribution histogram."""
        if self.dataset is None:
            self.load_dataset()
        
        text_lengths = [len(text) for text in self.dataset['text']]
        
        plt.figure(figsize=(12, 6))
        plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Text Lengths', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_length = statistics.mean(text_lengths)
        median_length = statistics.median(text_lengths)
        plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.0f}')
        plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.0f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'text_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_word_count_distribution_plot(self, output_dir: str) -> None:
        """Create word count distribution histogram."""
        if self.dataset is None:
            self.load_dataset()
        
        word_counts = [len(text.split()) for text in self.dataset['text']]
        
        plt.figure(figsize=(12, 6))
        plt.hist(word_counts, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        plt.title('Distribution of Word Counts', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_words = statistics.mean(word_counts)
        median_words = statistics.median(word_counts)
        plt.axvline(mean_words, color='red', linestyle='--', label=f'Mean: {mean_words:.0f}')
        plt.axvline(median_words, color='green', linestyle='--', label=f'Median: {median_words:.0f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'word_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_medical_terminology_plot(self, output_dir: str) -> None:
        """Create medical terminology analysis bar chart."""
        medical_data = self.stats['medical_terminology_analysis']
        top_terms = medical_data['most_common_medical_terms'][:15]
        
        if not top_terms:
            return
        
        terms, counts = zip(*top_terms)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(terms)), counts, color='coral', alpha=0.7)
        plt.xlabel('Medical Terms')
        plt.ylabel('Occurrences')
        plt.title('Top 15 Most Common Medical Terms', fontsize=16, fontweight='bold')
        plt.xticks(range(len(terms)), terms, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'medical_terminology_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_patient_distribution_plot(self, output_dir: str) -> None:
        """Create patient report distribution histogram."""
        if self.dataset is None:
            self.load_dataset()
        
        patient_counts = Counter(self.dataset['patient_id'])
        reports_per_patient = list(patient_counts.values())
        
        plt.figure(figsize=(12, 6))
        plt.hist(reports_per_patient, bins=20, alpha=0.7, color='gold', edgecolor='black')
        plt.xlabel('Number of Reports per Patient')
        plt.ylabel('Number of Patients')
        plt.title('Distribution of Reports per Patient', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_reports = statistics.mean(reports_per_patient)
        median_reports = statistics.median(reports_per_patient)
        plt.axvline(mean_reports, color='red', linestyle='--', label=f'Mean: {mean_reports:.2f}')
        plt.axvline(median_reports, color='green', linestyle='--', label=f'Median: {median_reports:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'patient_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Command line interface for descriptive statistics."""
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics for medical report dataset"
    )
    parser.add_argument(
        '--dataset-path',
        default='data/processed/raw_dataset',
        help='Path to the dataset (default: data/processed/raw_dataset)'
    )
    parser.add_argument(
        '--output',
        help='Output file path (default: descriptive_statistics.json)',
        default='descriptive_statistics.json'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'csv', 'excel'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print summary to console'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )
    parser.add_argument(
        '--create-visualizations',
        action='store_true',
        help='Create visualization plots'
    )
    parser.add_argument(
        '--viz-output-dir',
        default='visualizations',
        help='Directory to save visualization files (default: visualizations)'
    )
    parser.add_argument(
        '--num-reports',
        type=int,
        default=500,
        help='Number of reports to analyze (default: 500, same as translation subset)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible subset selection (default: 42, same as translation)'
    )
    parser.add_argument(
        '--all-reports',
        action='store_true',
        help='Analyze all reports instead of subset'
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = DescriptiveStatistics(dataset_path=args.dataset_path)
        
        # Determine subset parameters
        num_reports = None if args.all_reports else args.num_reports
        random_seed = None if args.all_reports else args.random_seed
        
        # Generate statistics
        print("Generating descriptive statistics...")
        stats = analyzer.get_comprehensive_statistics(num_reports, random_seed)
        
        # Print summary if requested
        if args.print_summary:
            analyzer.print_summary()
        
        # Save results if requested
        if not args.no_save:
            analyzer.save_results(args.output, args.format)
        
        # Create visualizations if requested
        if args.create_visualizations:
            analyzer.create_visualizations(args.viz_output_dir)
        
        print("Descriptive statistics generation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
