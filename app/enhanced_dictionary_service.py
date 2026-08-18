"""
Enhanced dictionary service for direct string matching from the enhanced dictionary file.

This service loads the enhanced dictionary with n-grams and provides direct string matching
capabilities for the --use-dictionary-only flag, with collision resolution between single
words and n-grams.
"""

import csv
import os
import re
from typing import Dict, Optional, List, Any, Tuple
from pathlib import Path


class EnhancedDictionaryService:
    """Service for direct string matching from the enhanced dictionary with n-grams."""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the enhanced dictionary service.
        
        Args:
            dictionary_path: Path to the enhanced dictionary CSV file. If None, uses default path.
        """
        if dictionary_path is None:
            # Default to enhanced_dictionary_filtered_qa_with_ngrams_final.csv
            project_root = Path(__file__).parent.parent
            dictionary_path = project_root / "assets" / "dictionaries" / "enhanced_dictionary_filtered_qa_with_ngrams_final.csv"
        
        self.dictionary_path = str(dictionary_path)
        self.dictionary = {}
        self.terms_by_length = {}  # Group terms by length for efficient longest-match-first lookup
        self.loaded = False
        
        # Load dictionary if file exists
        if os.path.exists(self.dictionary_path):
            self.load_dictionary()
    
    def load_dictionary(self) -> None:
        """
        Load the enhanced dictionary from CSV file.
        
        Raises:
            FileNotFoundError: If dictionary file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not os.path.exists(self.dictionary_path):
            raise FileNotFoundError(f"Dictionary file not found: {self.dictionary_path}")
        
        self.dictionary = {}
        self.terms_by_length = {}
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                
                for row in reader:
                    term = row.get('term', '').strip()
                    if not term:
                        continue
                    
                    # Create dictionary entry
                    entry = {
                        'term': term,
                        'total_count': int(row.get('total_count', 0)),
                        'translation_de': row.get('translation_de', '').strip(),
                        'sources': row.get('sources', '').strip(),
                        'verified': row.get('verified', 'FALSE').strip().upper() == 'TRUE'
                    }
                    
                    # Store under original term (case-insensitive)
                    term_key = term.lower()
                    self.dictionary[term_key] = entry
                    
                    # Group by word count for efficient longest-match-first lookup
                    word_count = len(term.split())
                    if word_count not in self.terms_by_length:
                        self.terms_by_length[word_count] = []
                    self.terms_by_length[word_count].append(term_key)
            
            self.loaded = True
            print(f"Loaded enhanced dictionary with {len(self.dictionary)} entries from {self.dictionary_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading enhanced dictionary from {self.dictionary_path}: {e}")
    
    def find_matches_in_text(self, text: str, target_lang: str = 'de') -> List[Dict[str, Any]]:
        """
        Find all dictionary matches in the given text using longest-match-first strategy.
        
        Args:
            text: The text to search for matches
            target_lang: Target language code (currently only 'de' supported)
            
        Returns:
            List of matches with their positions and translations
        """
        if not self.loaded:
            return []
        
        matches = []
        text_lower = text.lower()
        
        # Get all term lengths, sorted in descending order (longest first)
        term_lengths = sorted(self.terms_by_length.keys(), reverse=True)
        
        # Track which positions have already been matched to avoid overlaps
        matched_positions = set()
        
        for word_count in term_lengths:
            for term_key in self.terms_by_length[word_count]:
                entry = self.dictionary[term_key]
                
                # Skip if no translation available
                if not entry['translation_de']:
                    continue
                
                # Find all occurrences of this term in the text
                pattern = re.escape(term_key)
                
                for match in re.finditer(pattern, text_lower):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Check if this position overlaps with any already matched position
                    if any(start_pos < pos < end_pos or start_pos < pos + 1 < end_pos 
                           for pos in matched_positions):
                        continue
                    
                    # Add this match
                    matches.append({
                        'term': entry['term'],
                        'translation': entry['translation_de'],
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'verified': entry['verified'],
                        'sources': entry['sources'],
                        'total_count': entry['total_count'],
                        'word_count': word_count
                    })
                    
                    # Mark these positions as matched
                    for pos in range(start_pos, end_pos):
                        matched_positions.add(pos)
        
        # Sort matches by position in text
        matches.sort(key=lambda x: x['start_pos'])
        
        return matches
    
    def get_translation_for_term(self, term: str, target_lang: str = 'de') -> Optional[str]:
        """
        Get translation for a specific term.
        
        Args:
            term: The term to translate
            target_lang: Target language code (currently only 'de' supported)
            
        Returns:
            Translation if found, None otherwise
        """
        if not self.loaded:
            return None
        
        term_key = term.lower().strip()
        if term_key in self.dictionary:
            entry = self.dictionary[term_key]
            return entry['translation_de'] if entry['translation_de'] else None
        
        return None
    
    def get_translation_with_metadata(self, term: str, target_lang: str = 'de') -> Optional[Dict[str, Any]]:
        """
        Get translation with metadata for a specific term.
        
        Args:
            term: The term to translate
            target_lang: Target language code (currently only 'de' supported)
            
        Returns:
            Dictionary with translation and metadata if found, None otherwise
        """
        if not self.loaded:
            return None
        
        term_key = term.lower().strip()
        if term_key in self.dictionary:
            entry = self.dictionary[term_key]
            if entry['translation_de']:
                return {
                    'translation': entry['translation_de'],
                    'source': 'enhanced_dictionary',
                    'verified': entry['verified'],
                    'sources': entry['sources'],
                    'total_count': entry['total_count'],
                    'term': entry['term']
                }
        
        return None
    
    def is_available(self) -> bool:
        """
        Check if the dictionary is loaded and available.
        
        Returns:
            True if dictionary is loaded, False otherwise
        """
        return self.loaded and len(self.dictionary) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dictionary.
        
        Returns:
            Dictionary with statistics
        """
        if not self.loaded:
            return {'loaded': False, 'total_entries': 0}
        
        total_entries = len(self.dictionary)
        verified_entries = sum(1 for entry in self.dictionary.values() if entry['verified'])
        
        # Count entries by word count
        entries_by_word_count = {}
        for word_count, terms in self.terms_by_length.items():
            entries_by_word_count[word_count] = len(terms)
        
        return {
            'loaded': True,
            'total_entries': total_entries,
            'verified_entries': verified_entries,
            'entries_by_word_count': entries_by_word_count,
            'dictionary_path': self.dictionary_path
        }


# Global enhanced dictionary instance
_enhanced_dictionary = None


def get_enhanced_dictionary(dictionary_path: Optional[str] = None) -> EnhancedDictionaryService:
    """
    Get the global enhanced dictionary instance.
    
    Args:
        dictionary_path: Path to the enhanced dictionary CSV file. If None, uses default path.
        
    Returns:
        EnhancedDictionaryService instance
    """
    global _enhanced_dictionary
    
    if _enhanced_dictionary is None:
        _enhanced_dictionary = EnhancedDictionaryService(dictionary_path)
    
    return _enhanced_dictionary


def find_dictionary_matches_in_text(text: str, target_lang: str = 'de', dictionary_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to find dictionary matches in text.
    
    Args:
        text: The text to search for matches
        target_lang: Target language code (currently only 'de' supported)
        dictionary_path: Path to the enhanced dictionary CSV file. If None, uses default path.
        
    Returns:
        List of matches with their positions and translations
    """
    dictionary = get_enhanced_dictionary(dictionary_path)
    return dictionary.find_matches_in_text(text, target_lang)
