#!/usr/bin/env python3
"""
Command line interface for dictionary management.

This tool provides commands to manage the concept dictionary, including
viewing stats, searching terms, and testing translations.
"""

import argparse
import sys
from pathlib import Path
from dictionary_service import get_concept_dictionary, ConceptDictionary


def show_stats(dictionary_path: str = None):
    """Show dictionary statistics."""
    try:
        dictionary = get_concept_dictionary(dictionary_path)
        stats = dictionary.get_stats()
        
        if not stats['loaded']:
            print("❌ Dictionary not loaded")
            return
        
        print("📊 Dictionary Statistics")
        print("=" * 30)
        print(f"Total entries: {stats['total_entries']}")
        print(f"Verified entries: {stats['verified_entries']}")
        print(f"Dictionary path: {stats['dictionary_path']}")
        
        print(f"\nTranslations by language:")
        for lang, count in stats['translations_by_language'].items():
            print(f"  {lang.upper()}: {count}")
        
    except Exception as e:
        print(f"❌ Error loading dictionary: {e}")


def search_terms(query: str, limit: int = 10, dictionary_path: str = None):
    """Search for terms in the dictionary."""
    try:
        dictionary = get_concept_dictionary(dictionary_path)
        
        if not dictionary.is_available():
            print("❌ Dictionary not available")
            return
        
        matches = dictionary.search_terms(query, limit)
        
        if not matches:
            print(f"❌ No matches found for '{query}'")
            return
        
        print(f"🔍 Search results for '{query}' (showing {len(matches)} results)")
        print("=" * 60)
        
        for i, match in enumerate(matches, 1):
            verified = "✅" if match['verified'] else "❌"
            print(f"{i:2d}. {verified} {match['term']}")
            print(f"    Normalized: {match['normalized_text']}")
            print(f"    Categories: {match['categories']}")
            print(f"    Translations:")
            for lang, translation in match['translations'].items():
                if translation and not translation.startswith(f'[{lang.upper()}_TRANSLATION_NEEDED]'):
                    print(f"      {lang.upper()}: {translation}")
            print()
        
    except Exception as e:
        print(f"❌ Error searching dictionary: {e}")


def test_translation(term: str, target_lang: str, dictionary_path: str = None):
    """Test translation of a term."""
    try:
        dictionary = get_concept_dictionary(dictionary_path)
        
        if not dictionary.is_available():
            print("❌ Dictionary not available")
            return
        
        result = dictionary.get_translation_with_metadata(term, target_lang)
        
        if result:
            verified = "✅" if result['verified'] else "❌"
            print(f"✅ Translation found for '{term}' → '{target_lang.upper()}'")
            print(f"Translation: {result['translation']}")
            print(f"Verified: {verified}")
            print(f"Categories: {result['categories']}")
            print(f"Failure count: {result['failure_count']}")
            print(f"Avg confidence: {result['avg_confidence']}")
        else:
            print(f"❌ No translation found for '{term}' → '{target_lang.upper()}'")
        
    except Exception as e:
        print(f"❌ Error testing translation: {e}")


def main():
    """Command line interface for dictionary management."""
    parser = argparse.ArgumentParser(
        description="Manage concept dictionary for medical term translations"
    )
    parser.add_argument(
        '--dictionary-path',
        help='Path to the dictionary CSV file (default: assets/dictionaries/concept_dictionary.csv)',
        default=None
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show dictionary statistics')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for terms in dictionary')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, default=10, help='Maximum number of results')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test translation of a term')
    test_parser.add_argument('term', help='Term to translate')
    test_parser.add_argument('target_lang', help='Target language code (de, fr, es, it)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'stats':
        show_stats(args.dictionary_path)
    elif args.command == 'search':
        search_terms(args.query, args.limit, args.dictionary_path)
    elif args.command == 'test':
        test_translation(args.term, args.target_lang, args.dictionary_path)
    
    return 0


if __name__ == '__main__':
    exit(main())
