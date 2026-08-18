"""
Dictionary service for concept translations.

This service loads and queries a CSV dictionary file containing concept-translation pairs
to provide fallback translations when SNOMED CT translation fails.
"""

import csv
import os
from typing import Dict, Optional, List, Any
from pathlib import Path


class ConceptDictionary:
    """Service for loading and querying concept translations from a CSV dictionary."""
    
    def __init__(self, dictionary_path: Optional[str] = None):
        """
        Initialize the concept dictionary service.
        
        Args:
            dictionary_path: Path to the CSV dictionary file. If None, uses default path.
        """
        if dictionary_path is None:
            # Default to concept_dictionary.csv in the assets/dictionaries directory
            project_root = Path(__file__).parent.parent
            dictionary_path = project_root / "assets" / "dictionaries" / "concept_dictionary.csv"
        
        self.dictionary_path = str(dictionary_path)
        self.dictionary = {}
        self.loaded = False
        
        # Load dictionary if file exists
        if os.path.exists(self.dictionary_path):
            self.load_dictionary()
    
    def load_dictionary(self) -> None:
        """
        Load the concept dictionary from CSV file.
        
        Raises:
            FileNotFoundError: If dictionary file doesn't exist
            ValueError: If CSV format is invalid
        """
        if not os.path.exists(self.dictionary_path):
            raise FileNotFoundError(f"Dictionary file not found: {self.dictionary_path}")
        
        self.dictionary = {}
        
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    term = row.get('term', '').strip()
                    if not term:
                        continue
                    
                    # Store only the original term as key
                    
                    # Create dictionary entry
                    entry = {
                        'term': term,
                        'categories': row.get('categories', '').strip(),
                        'failure_count': int(row.get('failure_count', 0)),
                        'avg_confidence': float(row.get('avg_confidence', 0)),
                        'translations': {
                            'de': row.get('translation_de', '').strip(),
                            'fr': row.get('translation_fr', '').strip(),
                            'es': row.get('translation_es', '').strip(),
                            'it': row.get('translation_it', '').strip()
                        },
                        'notes': row.get('notes', '').strip(),
                        'verified': row.get('verified', 'FALSE').strip().upper() == 'TRUE'
                    }
                    
                    # Store under term only (case-insensitive)
                    term_key = term.lower()
                    self.dictionary[term_key] = entry
            
            self.loaded = True
            print(f"Loaded concept dictionary with {len(self.dictionary)} entries from {self.dictionary_path}")
            
        except Exception as e:
            raise ValueError(f"Error loading dictionary from {self.dictionary_path}: {e}")
    
    def get_translation(self, term: str, target_lang: str) -> Optional[str]:
        """
        Get translation for a term in the target language.
        Uses the original exact string for lookup, not the normalized version.
        
        Args:
            term: The term to translate (original exact string)
            target_lang: Target language code (de, fr, es, it)
            
        Returns:
            Translation if found, None otherwise
        """
        if not self.loaded:
            return None
        
        # Use the original exact term for lookup (case-insensitive)
        term_key = term.lower().strip()
        
        # Try exact match with original term first
        if term_key in self.dictionary:
            entry = self.dictionary[term_key]
            translation = entry['translations'].get(target_lang, '').strip()
            
            # Only return verified translations or non-placeholder translations
            if translation and not translation.startswith('[DE_TRANSLATION_NEEDED]') and not translation.startswith('[FR_TRANSLATION_NEEDED]') and not translation.startswith('[ES_TRANSLATION_NEEDED]') and not translation.startswith('[IT_TRANSLATION_NEEDED]'):
                return translation
        
        return None
    
    def get_translation_with_metadata(self, term: str, target_lang: str) -> Optional[Dict[str, Any]]:
        """
        Get translation with metadata for a term in the target language.
        Uses the original exact string for lookup, not the normalized version.
        
        Args:
            term: The term to translate (original exact string)
            target_lang: Target language code (de, fr, es, it)
            
        Returns:
            Dictionary with translation and metadata if found, None otherwise
        """
        if not self.loaded:
            return None
        
        # Use the original exact term for lookup (case-insensitive)
        term_key = term.lower().strip()
        
        if term_key in self.dictionary:
            entry = self.dictionary[term_key]
            translation = entry['translations'].get(target_lang, '').strip()
            
            # Only return verified translations or non-placeholder translations
            if translation and not translation.startswith('[DE_TRANSLATION_NEEDED]') and not translation.startswith('[FR_TRANSLATION_NEEDED]') and not translation.startswith('[ES_TRANSLATION_NEEDED]') and not translation.startswith('[IT_TRANSLATION_NEEDED]'):
                return {
                    'translation': translation,
                    'source': 'dictionary',
                    'verified': entry['verified'],
                    'categories': entry['categories'],
                    'failure_count': entry['failure_count'],
                    'avg_confidence': entry['avg_confidence'],
                    'notes': entry['notes']
                }
        
        return None
    
    def get_verified_translation_with_metadata(self, term: str, target_lang: str) -> Optional[Dict[str, Any]]:
        """
        Get translation with metadata for a term in the target language, but only if verified.
        Uses the original exact string for lookup, not the normalized version.
        
        Args:
            term: The term to translate (original exact string)
            target_lang: Target language code (de, fr, es, it)
            
        Returns:
            Dictionary with translation and metadata if found and verified, None otherwise
        """
        if not self.loaded:
            return None
        
        # Use the original exact term for lookup (case-insensitive)
        term_key = term.lower().strip()
        
        if term_key in self.dictionary:
            entry = self.dictionary[term_key]
            translation = entry['translations'].get(target_lang, '').strip()
            
            # Only return verified translations or non-placeholder translations
            if (translation and 
                entry['verified'] and 
                not translation.startswith('[DE_TRANSLATION_NEEDED]') and 
                not translation.startswith('[FR_TRANSLATION_NEEDED]') and 
                not translation.startswith('[ES_TRANSLATION_NEEDED]') and 
                not translation.startswith('[IT_TRANSLATION_NEEDED]')):
                return {
                    'translation': translation,
                    'source': 'dictionary',
                    'verified': entry['verified'],
                    'categories': entry['categories'],
                    'failure_count': entry['failure_count'],
                    'avg_confidence': entry['avg_confidence'],
                    'notes': entry['notes']
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
        
        # Count translations by language
        translations_by_lang = {'de': 0, 'fr': 0, 'es': 0, 'it': 0}
        for entry in self.dictionary.values():
            for lang in translations_by_lang:
                translation = entry['translations'].get(lang, '').strip()
                if translation and not translation.startswith(f'[{lang.upper()}_TRANSLATION_NEEDED]'):
                    translations_by_lang[lang] += 1
        
        return {
            'loaded': True,
            'total_entries': total_entries,
            'verified_entries': verified_entries,
            'translations_by_language': translations_by_lang,
            'dictionary_path': self.dictionary_path
        }
    
    def search_terms(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for terms in the dictionary.
        Prioritizes exact matches with original terms over normalized text.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching dictionary entries
        """
        if not self.loaded:
            return []
        
        query_lower = query.lower()
        matches = []
        
        for term_key, entry in self.dictionary.items():
            # Check if this is an exact match with the original term
            is_exact_original_match = (term_key == query_lower and term_key == entry['term'].lower())
            # Check if query is contained in original term
            is_contained_match = (query_lower in term_key)
            
            if is_exact_original_match or is_contained_match:
                matches.append({
                    'term': entry['term'],
                    'categories': entry['categories'],
                    'verified': entry['verified'],
                    'translations': entry['translations'],
                    'is_exact_match': is_exact_original_match
                })
        
        # Sort by exact match first, then verified status, then failure count
        matches.sort(key=lambda x: (not x['is_exact_match'], not x['verified'], -entry['failure_count']))
        
        return matches[:limit]


# Global dictionary instance
_concept_dictionary = None


def get_concept_dictionary(dictionary_path: Optional[str] = None) -> ConceptDictionary:
    """
    Get the global concept dictionary instance.
    
    Args:
        dictionary_path: Path to the CSV dictionary file. If None, uses default path.
        
    Returns:
        ConceptDictionary instance
    """
    global _concept_dictionary
    
    if _concept_dictionary is None:
        _concept_dictionary = ConceptDictionary(dictionary_path)
    
    return _concept_dictionary


def translate_with_dictionary(term: str, target_lang: str, dictionary_path: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to translate a term using the dictionary.
    
    Args:
        term: The term to translate
        target_lang: Target language code (de, fr, es, it)
        dictionary_path: Path to the CSV dictionary file. If None, uses default path.
        
    Returns:
        Translation if found, None otherwise
    """
    dictionary = get_concept_dictionary(dictionary_path)
    return dictionary.get_translation(term, target_lang)


def translate_with_dictionary_metadata(term: str, target_lang: str, dictionary_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to translate a term using the dictionary with metadata.
    
    Args:
        term: The term to translate
        target_lang: Target language code (de, fr, es, it)
        dictionary_path: Path to the CSV dictionary file. If None, uses default path.
        
    Returns:
        Dictionary with translation and metadata if found, None otherwise
    """
    dictionary = get_concept_dictionary(dictionary_path)
    return dictionary.get_translation_with_metadata(term, target_lang)
