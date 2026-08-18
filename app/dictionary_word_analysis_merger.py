"""
Dictionary and Word Analysis Merger

This script merges the existing merged_dictionary.csv with word analysis outputs
to create an enhanced dictionary that includes frequency information and n-grams.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import Counter
import pandas as pd


class DictionaryWordAnalysisMerger:
    """Merges dictionary data with word analysis results."""
    
    def __init__(self):
        self.merged_dictionary = {}
        self.word_frequencies = {}
        self.ngram_frequencies = {}
        self.failed_translations = {}
    
    def load_merged_dictionary(self, file_path: str) -> None:
        """
        Load the existing merged dictionary.
        
        Args:
            file_path: Path to the merged_dictionary.csv file
        """
        print(f"Loading merged dictionary from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                term = row['term'].lower().strip()
                self.merged_dictionary[term] = row
        
        print(f"Loaded {len(self.merged_dictionary)} terms from merged dictionary")
    
    def load_word_analysis_results(self, word_freq_file: str, ngram_freq_file: str = None, 
                                 failed_translations_file: str = None) -> None:
        """
        Load word analysis results from JSON files.
        
        Args:
            word_freq_file: Path to word frequency JSON file
            ngram_freq_file: Path to n-gram frequency JSON file (optional)
            failed_translations_file: Path to failed translations JSON file (optional)
        """
        # Load word frequencies
        if Path(word_freq_file).exists():
            print(f"Loading word frequencies from: {word_freq_file}")
            with open(word_freq_file, 'r', encoding='utf-8') as f:
                word_data = json.load(f)
                for item in word_data:
                    word = item['word'].lower().strip()
                    self.word_frequencies[word] = item['frequency']
            print(f"Loaded {len(self.word_frequencies)} word frequencies")
        else:
            print(f"Word frequency file not found: {word_freq_file}")
        
        # Load n-gram frequencies
        if ngram_freq_file and Path(ngram_freq_file).exists():
            print(f"Loading n-gram frequencies from: {ngram_freq_file}")
            with open(ngram_freq_file, 'r', encoding='utf-8') as f:
                ngram_data = json.load(f)
                for ngram_size, ngrams in ngram_data.items():
                    self.ngram_frequencies[ngram_size] = {}
                    for item in ngrams:
                        ngram = item['ngram'].lower().strip()
                        self.ngram_frequencies[ngram_size][ngram] = item['frequency']
            print(f"Loaded n-gram frequencies for sizes: {list(self.ngram_frequencies.keys())}")
        else:
            print(f"N-gram frequency file not found: {ngram_freq_file}")
        
        # Load failed translations
        if failed_translations_file and Path(failed_translations_file).exists():
            print(f"Loading failed translations from: {failed_translations_file}")
            with open(failed_translations_file, 'r', encoding='utf-8') as f:
                failed_data = json.load(f)
                for item in failed_data.get('failed_translations', []):
                    term = item['term'].lower().strip()
                    if term not in self.failed_translations:
                        self.failed_translations[term] = []
                    self.failed_translations[term].append(item)
            print(f"Loaded {len(self.failed_translations)} failed translation terms")
        else:
            print(f"Failed translations file not found: {failed_translations_file}")
    
    def extract_terms_from_ngrams(self) -> Set[str]:
        """
        Extract individual terms from n-grams for frequency analysis.
        
        Returns:
            Set of individual terms found in n-grams
        """
        terms = set()
        for ngram_size, ngrams in self.ngram_frequencies.items():
            for ngram in ngrams.keys():
                # Split n-gram into individual words
                words = ngram.split()
                terms.update(words)
        return terms
    
    def calculate_term_frequency_score(self, term: str) -> float:
        """
        Calculate a frequency score for a term based on word and n-gram frequencies.
        
        Args:
            term: Term to calculate score for
            
        Returns:
            Frequency score (0.0 to 1.0)
        """
        score = 0.0
        
        # Direct word frequency
        if term in self.word_frequencies:
            score += 0.7  # Base score for direct word match
        
        # N-gram frequency (term appears in n-grams)
        ngram_score = 0.0
        for ngram_size, ngrams in self.ngram_frequencies.items():
            for ngram, freq in ngrams.items():
                if term in ngram.split():
                    # Weight by n-gram size (bigrams get higher weight than trigrams)
                    weight = 1.0 / (int(ngram_size.split('-')[0]) - 1) if int(ngram_size.split('-')[0]) > 1 else 1.0
                    ngram_score += freq * weight * 0.1  # Scale down n-gram contribution
        
        score += min(ngram_score, 0.3)  # Cap n-gram contribution at 0.3
        
        return min(score, 1.0)  # Cap total score at 1.0
    
    def create_enhanced_dictionary(self) -> List[Dict[str, Any]]:
        """
        Create an enhanced dictionary by appending new terms from word analysis to existing dictionary.
        Existing dictionary entries are kept unchanged, only new terms are added.
        
        Returns:
            List of enhanced dictionary entries
        """
        print("Creating enhanced dictionary...")
        
        enhanced_entries = []
        existing_terms = set()
        
        # Keep all existing dictionary entries unchanged
        print("Preserving existing dictionary entries...")
        for term, entry in self.merged_dictionary.items():
            enhanced_entries.append(entry.copy())
            existing_terms.add(term.lower().strip())
        
        # Store original column names for consistency
        original_columns = list(self.merged_dictionary.values())[0].keys() if self.merged_dictionary else []
        
        print(f"Preserved {len(enhanced_entries)} existing dictionary entries")
        
        # Add high-frequency terms from word analysis that aren't in the dictionary
        print("Adding new high-frequency terms from word analysis...")
        high_freq_threshold = 5  # Only add terms that appear at least 5 times
        new_terms_added = 0
        
        for term, freq in self.word_frequencies.items():
            term_lower = term.lower().strip()
            if term_lower not in existing_terms and freq >= high_freq_threshold:
                # Create new entry for high-frequency term using original column structure
                enhanced_entry = {
                    'term': term,
                    'normalized_text': term.title(),
                    'categories': 'WordAnalysis',
                    'success_count': 0,
                    'failure_count': 0,
                    'total_count': freq,  # Store frequency in total_count
                    'avg_confidence': 0.0,
                    'translation_de': '',
                    'translation_fr': '',
                    'translation_es': '',
                    'translation_it': '',
                    'sources': 'analysis',
                    'file_count': 0,
                    'notes': f'High frequency term from word analysis: {freq} occurrences',
                    'verified': False
                }
                
                enhanced_entries.append(enhanced_entry)
                new_terms_added += 1
        
        print(f"Added {new_terms_added} new terms from word analysis")
        
        # Add high-frequency n-grams that aren't already covered by existing terms
        print("Adding new high-frequency n-grams...")
        ngram_threshold = 3  # Only add n-grams that appear at least 3 times
        new_ngrams_added = 0
        
        for ngram_size, ngrams in self.ngram_frequencies.items():
            for ngram, freq in ngrams.items():
                if freq >= ngram_threshold:
                    # Check if ALL words in the n-gram are already in the dictionary
                    ngram_words = ngram.split()
                    all_words_exist = all(word.lower().strip() in existing_terms for word in ngram_words)
                    
                    # Only exclude if ALL words already exist (less restrictive)
                    if not all_words_exist:
                        # Create new entry for high-frequency n-gram using original column structure
                        enhanced_entry = {
                            'term': ngram,
                            'normalized_text': ngram.title(),
                            'categories': f'WordAnalysis-{ngram_size}',
                            'success_count': 0,
                            'failure_count': 0,
                            'total_count': freq,  # Store frequency in total_count
                            'avg_confidence': 0.0,
                            'translation_de': '',
                            'translation_fr': '',
                            'translation_es': '',
                            'translation_it': '',
                            'sources': 'analysis',
                            'file_count': 0,
                            'notes': f'High frequency {ngram_size} from word analysis: {freq} occurrences',
                            'verified': False
                        }
                        
                        enhanced_entries.append(enhanced_entry)
                        new_ngrams_added += 1
        
        print(f"Added {new_ngrams_added} new n-grams from word analysis")
        
        # Sort by frequency score (descending), but keep original entries first
        original_entries = enhanced_entries[:len(self.merged_dictionary)]
        new_entries = enhanced_entries[len(self.merged_dictionary):]
        new_entries.sort(key=lambda x: x.get('frequency_score', 0), reverse=True)
        enhanced_entries = original_entries + new_entries
        
        print(f"Created enhanced dictionary with {len(enhanced_entries)} total entries")
        print(f"  - {len(original_entries)} original dictionary entries (unchanged)")
        print(f"  - {len(new_entries)} new entries from word analysis")
        
        return enhanced_entries
    
    def save_enhanced_dictionary(self, enhanced_entries: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save the enhanced dictionary to a CSV file using original column structure.
        
        Args:
            enhanced_entries: List of enhanced dictionary entries
            output_path: Path to save the enhanced dictionary
        """
        print(f"Saving enhanced dictionary to: {output_path}")
        
        if not enhanced_entries:
            print("No entries to save")
            return
        
        # Use the original column structure from merged_dictionary
        original_columns = [
            'term', 'normalized_text', 'categories', 'success_count', 'failure_count', 
            'total_count', 'avg_confidence', 'translation_de', 'translation_fr', 
            'translation_es', 'translation_it', 'sources', 'file_count', 'notes', 'verified'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=original_columns)
            writer.writeheader()
            
            # Write entries with only original columns
            for entry in enhanced_entries:
                filtered_entry = {col: entry.get(col, '') for col in original_columns}
                writer.writerow(filtered_entry)
        
        print(f"Saved {len(enhanced_entries)} entries to {output_path}")
    
    def generate_summary_report(self, enhanced_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary report of the enhanced dictionary.
        
        Args:
            enhanced_entries: List of enhanced dictionary entries
            
        Returns:
            Dictionary containing summary statistics
        """
        total_entries = len(enhanced_entries)
        original_entries = sum(1 for entry in enhanced_entries if entry.get('sources') != 'analysis')
        new_entries = total_entries - original_entries
        
        # Count new entries by type
        new_word_entries = sum(1 for entry in enhanced_entries if entry.get('sources') == 'analysis' and entry.get('categories') == 'WordAnalysis')
        new_ngram_entries = sum(1 for entry in enhanced_entries if entry.get('sources') == 'analysis' and entry.get('categories', '').startswith('WordAnalysis-'))
        
        high_freq_terms = sum(1 for entry in enhanced_entries if int(entry.get('total_count', 0)) >= 10)
        failed_terms = sum(1 for entry in enhanced_entries if int(entry.get('failure_count', 0)) > 0)
        
        # Top new terms by total count (only from analysis)
        new_entries_only = [entry for entry in enhanced_entries if entry.get('sources') == 'analysis']
        top_freq_terms = sorted(new_entries_only, key=lambda x: int(x.get('total_count', 0)), reverse=True)[:10]
        
        # Top terms by success count (only from analysis)
        top_success_terms = sorted(new_entries_only, key=lambda x: int(x.get('success_count', 0)), reverse=True)[:10]
        
        return {
            'total_entries': total_entries,
            'original_entries': original_entries,
            'new_entries_from_word_analysis': new_entries,
            'new_word_entries': new_word_entries,
            'new_ngram_entries': new_ngram_entries,
            'high_frequency_terms': high_freq_terms,
            'terms_with_failed_translations': failed_terms,
            'top_frequency_terms': [(entry['term'], int(entry.get('total_count', 0))) for entry in top_freq_terms],
            'top_success_terms': [(entry['term'], int(entry.get('success_count', 0))) for entry in top_success_terms]
        }


def main():
    """Command line interface for dictionary and word analysis merger."""
    parser = argparse.ArgumentParser(
        description="Merge merged_dictionary.csv with word analysis outputs"
    )
    parser.add_argument(
        'merged_dictionary',
        help='Path to the merged_dictionary.csv file'
    )
    parser.add_argument(
        'word_frequency_file',
        help='Path to the word frequency JSON file from word analysis'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path for enhanced dictionary (default: enhanced_dictionary.csv)',
        default='enhanced_dictionary.csv'
    )
    parser.add_argument(
        '--ngram-file',
        help='Path to n-gram frequency JSON file (optional)'
    )
    parser.add_argument(
        '--failed-translations-file',
        help='Path to failed translations JSON file (optional)'
    )
    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='Print summary report to console'
    )
    parser.add_argument(
        '--save-summary',
        help='Save summary report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Create merger
    merger = DictionaryWordAnalysisMerger()
    
    try:
        # Load data
        merger.load_merged_dictionary(args.merged_dictionary)
        merger.load_word_analysis_results(
            args.word_frequency_file,
            args.ngram_file,
            args.failed_translations_file
        )
        
        # Create enhanced dictionary
        enhanced_entries = merger.create_enhanced_dictionary()
        
        # Save enhanced dictionary
        merger.save_enhanced_dictionary(enhanced_entries, args.output)
        
        # Generate and display summary
        summary = merger.generate_summary_report(enhanced_entries)
        
        if args.print_summary:
            print("\n" + "="*60)
            print("ENHANCED DICTIONARY SUMMARY")
            print("="*60)
            print(f"Total entries: {summary['total_entries']}")
            print(f"Original entries (unchanged): {summary['original_entries']}")
            print(f"New entries from word analysis: {summary['new_entries_from_word_analysis']}")
            print(f"  - New word entries: {summary['new_word_entries']}")
            print(f"  - New n-gram entries: {summary['new_ngram_entries']}")
            print(f"High frequency terms (≥10 occurrences): {summary['high_frequency_terms']}")
            print(f"Terms with failed translations: {summary['terms_with_failed_translations']}")
            
            if summary['top_frequency_terms']:
                print(f"\nTop 10 new terms by total count:")
                for i, (term, count) in enumerate(summary['top_frequency_terms'], 1):
                    print(f"  {i:2d}. {term:<30} {count:>6}")
            else:
                print(f"\nNo new high-frequency terms found")
            
            if summary['top_success_terms']:
                print(f"\nTop 10 new terms by success count:")
                for i, (term, count) in enumerate(summary['top_success_terms'], 1):
                    print(f"  {i:2d}. {term:<30} {count:>6}")
            else:
                print(f"\nNo new terms with success counts found")
        
        if args.save_summary:
            with open(args.save_summary, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to: {args.save_summary}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
