"""
Word frequency and n-gram analysis module for completed translation JSON files.

This module reads translation results from JSON files and generates lists of the most common words
and n-grams, filtering out short words like articles and prepositions. It now includes support for
abbreviations (minimum word length of 2) and n-gram analysis (bigrams, trigrams, etc.).
"""

import json
import re
import argparse
from typing import List, Dict, Any, Optional, Set
from collections import Counter
from pathlib import Path
import csv


class WordAnalyzer:
    """Analyzes word frequency and n-grams in translation JSON files."""
    
    def __init__(self, min_word_length: int = 2, exclude_common_words: bool = True, 
                 include_ngrams: bool = True, max_ngram_size: int = 3):
        """
        Initialize the word analyzer.
        
        Args:
            min_word_length: Minimum length for words to be included in analysis
            exclude_common_words: Whether to exclude common stop words
            include_ngrams: Whether to include n-gram analysis
            max_ngram_size: Maximum size of n-grams to analyze (2=bigrams, 3=trigrams, etc.)
        """
        self.min_word_length = min_word_length
        self.exclude_common_words = exclude_common_words
        self.include_ngrams = include_ngrams
        self.max_ngram_size = max_ngram_size
        
        # Common stop words in multiple languages (articles, prepositions, conjunctions, etc.)
        self.stop_words = {
            # English
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
            'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
            
            # German
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'eines',
            'und', 'oder', 'aber', 'in', 'an', 'auf', 'zu', 'für', 'von', 'mit', 'bei', 'nach',
            'über', 'unter', 'zwischen', 'durch', 'ohne', 'gegen', 'um', 'vor', 'hinter', 'neben',
            'ist', 'sind', 'war', 'waren', 'bin', 'bist', 'habe', 'hast', 'hat', 'haben', 'hatt',
            'werde', 'wirst', 'wird', 'werden', 'wurde', 'wurden', 'kann', 'kannst', 'können',
            'könnt', 'konnte', 'konnten', 'soll', 'sollst', 'sollen', 'sollt', 'sollte', 'sollten',
            'muss', 'musst', 'müssen', 'müsst', 'musste', 'mussten', 'darf', 'darfst', 'dürfen',
            'dürft', 'durfte', 'durften', 'mag', 'magst', 'mögen', 'mögt', 'mochte', 'mochten',
            'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'mich', 'dich', 'ihn', 'sie',
            'es', 'uns', 'euch', 'sie', 'mein', 'dein', 'sein', 'ihr', 'sein', 'unser', 'euer',
            'ihr', 'meine', 'deine', 'seine', 'ihre', 'seine', 'unsere', 'eure', 'ihre',
            
            # French
            'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'dans', 'sur', 'avec', 'sans',
            'pour', 'par', 'vers', 'chez', 'entre', 'sous', 'devant', 'derrière', 'près', 'loin',
            'et', 'ou', 'mais', 'donc', 'or', 'ni', 'car', 'est', 'sont', 'était', 'étaient',
            'suis', 'es', 'sommes', 'êtes', 'ai', 'as', 'a', 'avons', 'avez', 'ont', 'avais',
            'avait', 'avions', 'aviez', 'avaient', 'serai', 'seras', 'sera', 'serons', 'serez',
            'seront', 'serais', 'serait', 'serions', 'seriez', 'seraient', 'je', 'tu', 'il',
            'elle', 'nous', 'vous', 'ils', 'elles', 'me', 'te', 'se', 'nous', 'vous', 'se',
            'mon', 'ma', 'mes', 'ton', 'ta', 'tes', 'son', 'sa', 'ses', 'notre', 'nos',
            'votre', 'vos', 'leur', 'leurs',
            
            # Spanish
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'con',
            'por', 'para', 'sobre', 'bajo', 'entre', 'sin', 'hasta', 'desde', 'hacia', 'durante',
            'y', 'o', 'pero', 'sino', 'aunque', 'porque', 'si', 'que', 'como', 'cuando', 'donde',
            'es', 'son', 'era', 'eran', 'soy', 'eres', 'somos', 'sois', 'fui', 'fuiste', 'fue',
            'fuimos', 'fuisteis', 'fueron', 'he', 'has', 'ha', 'hemos', 'habéis', 'han', 'había',
            'habías', 'habíamos', 'habíais', 'habían', 'seré', 'serás', 'será', 'seremos',
            'seréis', 'serán', 'yo', 'tú', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros',
            'vosotras', 'ellos', 'ellas', 'me', 'te', 'se', 'nos', 'os', 'se', 'mi', 'tu',
            'su', 'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra',
            'vuestros', 'vuestras', 'su', 'sus',
            
            # Italian
            'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'a', 'da', 'in',
            'con', 'su', 'per', 'tra', 'fra', 'sopra', 'sotto', 'davanti', 'dietro', 'vicino',
            'lontano', 'e', 'o', 'ma', 'però', 'quindi', 'perché', 'se', 'che', 'come', 'quando',
            'dove', 'è', 'sono', 'era', 'erano', 'sono', 'sei', 'siamo', 'siete', 'fui', 'fosti',
            'fu', 'fummo', 'foste', 'furono', 'ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno',
            'avevo', 'avevi', 'aveva', 'avevamo', 'avevate', 'avevano', 'sarò', 'sarai', 'sarà',
            'saremo', 'sarete', 'saranno', 'io', 'tu', 'lui', 'lei', 'noi', 'voi', 'loro',
            'mi', 'ti', 'si', 'ci', 'vi', 'si', 'mio', 'tuo', 'suo', 'nostro', 'vostro',
            'loro', 'mia', 'tua', 'sua', 'nostra', 'vostra', 'loro', 'miei', 'tuoi', 'suoi',
            'nostri', 'vostri', 'loro', 'mie', 'tue', 'sue', 'nostre', 'vostre', 'loro'
        }
    
    def load_translations(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load translation data from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing translations
            
        Returns:
            List of translation dictionaries
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            if not isinstance(translations, list):
                raise ValueError("JSON file should contain a list of translations")
            
            return translations
        except FileNotFoundError:
            raise FileNotFoundError(f"Translation file not found: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
    
    def extract_original_texts(self, translations: List[Dict[str, Any]]) -> List[str]:
        """
        Extract original texts from translation data.
        
        Args:
            translations: List of translation dictionaries
            
        Returns:
            List of original text strings
        """
        original_texts = []
        for translation in translations:
            if 'original_text' in translation and translation['original_text']:
                original_texts.append(translation['original_text'])
        return original_texts
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing punctuation and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def should_include_word(self, word: str) -> bool:
        """
        Determine if a word should be included in the analysis.
        
        Args:
            word: Word to check
            
        Returns:
            True if word should be included, False otherwise
        """
        # Check minimum length
        if len(word) < self.min_word_length:
            return False
        
        # Check if it's a stop word
        if self.exclude_common_words and word.lower() in self.stop_words:
            return False
        
        # Check if it's purely numeric
        if word.isdigit():
            return False
        
        return True
    
    def analyze_word_frequency(self, texts: List[str]) -> Counter:
        """
        Analyze word frequency in the given texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Counter object with word frequencies
        """
        word_counter = Counter()
        
        for text in texts:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Split into words and count
            words = cleaned_text.split()
            for word in words:
                word_lower = word.lower()
                if self.should_include_word(word_lower):
                    word_counter[word_lower] += 1
        
        return word_counter
    
    def extract_ngrams(self, words: List[str], n: int) -> List[str]:
        """
        Extract n-grams from a list of words.
        
        Args:
            words: List of words to extract n-grams from
            n: Size of n-grams to extract
            
        Returns:
            List of n-grams as strings
        """
        if n <= 0 or n > len(words):
            return []
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def should_include_ngram(self, ngram: str) -> bool:
        """
        Determine if an n-gram should be included in the analysis.
        
        Args:
            ngram: N-gram to check
            
        Returns:
            True if n-gram should be included, False otherwise
        """
        words = ngram.split()
        
        # Check if any word in the n-gram is too short
        if any(len(word) < self.min_word_length for word in words):
            return False
        
        # Check if any word in the n-gram is a stop word (if excluding stop words)
        if self.exclude_common_words:
            if any(word.lower() in self.stop_words for word in words):
                return False
        
        # Check if any word is purely numeric
        if any(word.isdigit() for word in words):
            return False
        
        return True
    
    def analyze_ngram_frequency(self, texts: List[str]) -> Dict[int, Counter]:
        """
        Analyze n-gram frequency in the given texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            Dictionary mapping n-gram size to Counter object with n-gram frequencies
        """
        ngram_counters = {}
        
        for n in range(2, self.max_ngram_size + 1):
            ngram_counters[n] = Counter()
        
        for text in texts:
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            # Split into words
            words = cleaned_text.split()
            
            # Extract n-grams for each size
            for n in range(2, self.max_ngram_size + 1):
                ngrams = self.extract_ngrams(words, n)
                for ngram in ngrams:
                    ngram_lower = ngram.lower()
                    if self.should_include_ngram(ngram_lower):
                        ngram_counters[n][ngram_lower] += 1
        
        return ngram_counters
    
    def get_most_common_ngrams(self, json_file_path: str, top_n: int = 100) -> Dict[int, List[tuple]]:
        """
        Get the most common n-grams from a translation JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing translations
            top_n: Number of top n-grams to return for each size
            
        Returns:
            Dictionary mapping n-gram size to list of tuples (ngram, frequency) sorted by frequency
        """
        # Load translations
        translations = self.load_translations(json_file_path)
        
        # Extract original texts
        original_texts = self.extract_original_texts(translations)
        
        if not original_texts:
            print("Warning: No original texts found in the translation file")
            return {}
        
        # Analyze n-gram frequency
        ngram_frequencies = self.analyze_ngram_frequency(original_texts)
        
        # Get most common n-grams for each size
        most_common_ngrams = {}
        for n, counter in ngram_frequencies.items():
            most_common_ngrams[n] = counter.most_common(top_n)
        
        return most_common_ngrams
    
    def get_most_common_words(self, json_file_path: str, top_n: int = 100) -> List[tuple]:
        """
        Get the most common words from a translation JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing translations
            top_n: Number of top words to return
            
        Returns:
            List of tuples (word, frequency) sorted by frequency (descending)
        """
        # Load translations
        translations = self.load_translations(json_file_path)
        
        # Extract original texts
        original_texts = self.extract_original_texts(translations)
        
        if not original_texts:
            print("Warning: No original texts found in the translation file")
            return []
        
        # Analyze word frequency
        word_frequencies = self.analyze_word_frequency(original_texts)
        
        # Get most common words
        most_common = word_frequencies.most_common(top_n)
        
        return most_common
    
    def get_default_output_path(self, input_file_path: str, format: str = 'json') -> str:
        """
        Generate a default output path in the same directory as the input file.
        
        Args:
            input_file_path: Path to the input JSON file
            format: Output format ('json', 'csv', 'txt')
            
        Returns:
            Default output path in the same directory as input file
        """
        input_path = Path(input_file_path)
        input_dir = input_path.parent
        input_stem = input_path.stem
        
        # Create output filename based on input filename and format
        output_filename = f"{input_stem}_word_frequency.{format}"
        return str(input_dir / output_filename)
    
    def get_failed_translations_output_path(self, input_file_path: str, format: str = 'json') -> str:
        """
        Generate a default output path for failed translations analysis.
        
        Args:
            input_file_path: Path to the input JSON file
            format: Output format ('json', 'csv', 'txt')
            
        Returns:
            Default output path for failed translations in the same directory as input file
        """
        input_path = Path(input_file_path)
        input_dir = input_path.parent
        input_stem = input_path.stem
        
        # Create output filename for failed translations
        output_filename = f"{input_stem}_failed_translations.{format}"
        return str(input_dir / output_filename)
    
    def extract_snomed_translation_data(self, translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract SNOMED translation data from translation results.
        
        Args:
            translations: List of translation dictionaries
            
        Returns:
            List of SNOMED translation data
        """
        snomed_data = []
        
        for translation in translations:
            if 'snomed_translations' in translation:
                for snomed_translation in translation['snomed_translations']:
                    # Add context information from the parent translation
                    snomed_translation_with_context = snomed_translation.copy()
                    snomed_translation_with_context.update({
                        'file_path': translation.get('file_path', ''),
                        'patient_id': translation.get('patient_id', ''),
                        'study_id': translation.get('study_id', ''),
                        'source_lang': translation.get('source_lang', ''),
                        'target_lang': translation.get('target_lang', ''),
                        'model_used': translation.get('model_used', ''),
                        'translation_mode': translation.get('translation_mode', '')
                    })
                    snomed_data.append(snomed_translation_with_context)
        
        return snomed_data
    
    def analyze_failed_translations(self, json_file_path: str) -> Dict[str, Any]:
        """
        Analyze concepts that had no translation found.
        
        Args:
            json_file_path: Path to the JSON file containing translations
            
        Returns:
            Dictionary containing analysis of failed translations
        """
        # Load translations
        translations = self.load_translations(json_file_path)
        
        # Extract SNOMED translation data
        snomed_data = self.extract_snomed_translation_data(translations)
        
        if not snomed_data:
            return {
                'total_concepts': 0,
                'failed_translations': [],
                'failed_by_category': {},
                'failed_by_source': {},
                'summary': 'No SNOMED translation data found'
            }
        
        # Analyze failed translations
        failed_translations = []
        failed_by_category = {}
        failed_by_source = {}
        
        for snomed_item in snomed_data:
            # Check if translation failed
            success = snomed_item.get('success', False)
            translation = snomed_item.get('translation')
            
            if not success or not translation:
                failed_item = {
                    'term': snomed_item.get('term', ''),
                    'normalized_text': snomed_item.get('normalized_text', ''),
                    'category': snomed_item.get('category', ''),
                    'confidence_score': snomed_item.get('confidence_score', 0),
                    'codes': snomed_item.get('codes', {}),
                    'snomed_lookup': snomed_item.get('snomed_lookup', {}),
                    'error': snomed_item.get('error', ''),
                    'file_path': snomed_item.get('file_path', ''),
                    'patient_id': snomed_item.get('patient_id', ''),
                    'study_id': snomed_item.get('study_id', ''),
                    'source_lang': snomed_item.get('source_lang', ''),
                    'target_lang': snomed_item.get('target_lang', ''),
                    'model_used': snomed_item.get('model_used', ''),
                    'translation_mode': snomed_item.get('translation_mode', '')
                }
                failed_translations.append(failed_item)
                
                # Categorize by medical category
                category = snomed_item.get('category', 'unknown')
                if category not in failed_by_category:
                    failed_by_category[category] = []
                failed_by_category[category].append(failed_item)
                
                # Categorize by SNOMED lookup source
                snomed_lookup = snomed_item.get('snomed_lookup', {})
                source = snomed_lookup.get('source', 'none')
                if source not in failed_by_source:
                    failed_by_source[source] = []
                failed_by_source[source].append(failed_item)
        
        # Count total concepts
        total_concepts = len(snomed_data)
        total_failed = len(failed_translations)
        success_rate = ((total_concepts - total_failed) / total_concepts * 100) if total_concepts > 0 else 0
        
        return {
            'total_concepts': total_concepts,
            'total_failed': total_failed,
            'success_rate': round(success_rate, 2),
            'failed_translations': failed_translations,
            'failed_by_category': failed_by_category,
            'failed_by_source': failed_by_source,
            'summary': f'{total_failed} out of {total_concepts} concepts failed translation ({100-success_rate:.1f}% failure rate)'
        }
    
    def get_failed_translations_summary(self, json_file_path: str) -> List[tuple]:
        """
        Get a summary of failed translations grouped by term.
        
        Args:
            json_file_path: Path to the JSON file containing translations
            
        Returns:
            List of tuples (term, failure_count, categories) sorted by failure count
        """
        analysis = self.analyze_failed_translations(json_file_path)
        failed_translations = analysis['failed_translations']
        
        # Group by term
        term_failures = {}
        for failed in failed_translations:
            term = failed['term']
            if term not in term_failures:
                term_failures[term] = {
                    'count': 0,
                    'categories': set(),
                    'normalized_text': failed.get('normalized_text', ''),
                    'confidence_scores': []
                }
            
            term_failures[term]['count'] += 1
            term_failures[term]['categories'].add(failed.get('category', 'unknown'))
            term_failures[term]['confidence_scores'].append(failed.get('confidence_score', 0))
        
        # Convert to list of tuples and sort by failure count
        summary = []
        for term, data in term_failures.items():
            avg_confidence = sum(data['confidence_scores']) / len(data['confidence_scores']) if data['confidence_scores'] else 0
            summary.append((
                term,
                data['count'],
                ', '.join(sorted(data['categories'])),
                data['normalized_text'],
                round(avg_confidence, 2)
            ))
        
        # Sort by failure count (descending)
        summary.sort(key=lambda x: x[1], reverse=True)
        
        return summary
    
    def save_results(self, results: List[tuple], output_path: str, format: str = 'json') -> None:
        """
        Save word frequency results to a file.
        
        Args:
            results: List of tuples (word, frequency)
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'txt')
        """
        if format == 'json':
            data = [{'word': word, 'frequency': freq} for word, freq in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['word', 'frequency'])
                writer.writerows(results)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for word, freq in results:
                    f.write(f"{word}: {freq}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_ngram_results(self, ngram_results: Dict[int, List[tuple]], output_path: str, format: str = 'json') -> None:
        """
        Save n-gram frequency results to a file.
        
        Args:
            ngram_results: Dictionary mapping n-gram size to list of tuples (ngram, frequency)
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'txt')
        """
        if format == 'json':
            data = {}
            for n, results in ngram_results.items():
                data[f'{n}-grams'] = [{'ngram': ngram, 'frequency': freq} for ngram, freq in results]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['ngram_size', 'ngram', 'frequency'])
                for n, results in ngram_results.items():
                    for ngram, freq in results:
                        writer.writerow([n, ngram, freq])
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for n, results in ngram_results.items():
                    f.write(f"\n{n}-grams:\n")
                    f.write("-" * 40 + "\n")
                    for ngram, freq in results:
                        f.write(f"{ngram}: {freq}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_failed_translations_analysis(self, analysis: Dict[str, Any], output_path: str, format: str = 'json') -> None:
        """
        Save failed translations analysis to a file.
        
        Args:
            analysis: Dictionary containing failed translations analysis
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'txt')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'term', 'normalized_text', 'category', 'confidence_score', 
                    'file_path', 'patient_id', 'study_id', 'source_lang', 
                    'target_lang', 'model_used', 'translation_mode', 'error'
                ])
                
                for failed in analysis['failed_translations']:
                    writer.writerow([
                        failed.get('term', ''),
                        failed.get('normalized_text', ''),
                        failed.get('category', ''),
                        failed.get('confidence_score', 0),
                        failed.get('file_path', ''),
                        failed.get('patient_id', ''),
                        failed.get('study_id', ''),
                        failed.get('source_lang', ''),
                        failed.get('target_lang', ''),
                        failed.get('model_used', ''),
                        failed.get('translation_mode', ''),
                        failed.get('error', '')
                    ])
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Failed Translations Analysis\n")
                f.write(f"==========================\n\n")
                f.write(f"Summary: {analysis['summary']}\n")
                f.write(f"Total concepts: {analysis['total_concepts']}\n")
                f.write(f"Failed translations: {analysis['total_failed']}\n")
                f.write(f"Success rate: {analysis['success_rate']}%\n\n")
                
                f.write(f"Failed by Category:\n")
                f.write(f"------------------\n")
                for category, items in analysis['failed_by_category'].items():
                    f.write(f"{category}: {len(items)} failures\n")
                
                f.write(f"\nFailed by Source:\n")
                f.write(f"----------------\n")
                for source, items in analysis['failed_by_source'].items():
                    f.write(f"{source}: {len(items)} failures\n")
                
                f.write(f"\nDetailed Failed Translations:\n")
                f.write(f"----------------------------\n")
                for failed in analysis['failed_translations']:
                    f.write(f"Term: {failed.get('term', '')}\n")
                    f.write(f"Category: {failed.get('category', '')}\n")
                    f.write(f"Confidence: {failed.get('confidence_score', 0)}\n")
                    f.write(f"File: {failed.get('file_path', '')}\n")
                    f.write(f"Error: {failed.get('error', 'No specific error')}\n")
                    f.write(f"---\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def save_failed_translations_summary(self, summary: List[tuple], output_path: str, format: str = 'json') -> None:
        """
        Save failed translations summary to a file.
        
        Args:
            summary: List of tuples (term, failure_count, categories, normalized_text, avg_confidence)
            output_path: Path to save the results
            format: Output format ('json', 'csv', 'txt')
        """
        if format == 'json':
            data = [
                {
                    'term': term,
                    'failure_count': count,
                    'categories': categories,
                    'normalized_text': normalized_text,
                    'avg_confidence': avg_confidence
                }
                for term, count, categories, normalized_text, avg_confidence in summary
            ]
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['term', 'failure_count', 'categories', 'normalized_text', 'avg_confidence'])
                writer.writerows(summary)
        
        elif format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Failed Translations Summary\n")
                f.write("==========================\n\n")
                for i, (term, count, categories, normalized_text, avg_confidence) in enumerate(summary, 1):
                    f.write(f"{i:3d}. {term:<25} (failed {count} times)\n")
                    f.write(f"     Categories: {categories}\n")
                    f.write(f"     Normalized: {normalized_text}\n")
                    f.write(f"     Avg Confidence: {avg_confidence}\n\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def main():
    """Command line interface for word frequency and n-gram analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze word frequency and n-grams in completed translation JSON files"
    )
    parser.add_argument(
        'input_file',
        help='Path to the translation JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: same directory as input file with _word_frequency suffix)',
        default=None
    )
    parser.add_argument(
        '-f', '--format',
        choices=['json', 'csv', 'txt'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '-n', '--top-n',
        type=int,
        default=100,
        help='Number of top words to return (default: 100)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=2,
        help='Minimum word length to include (default: 2)'
    )
    parser.add_argument(
        '--no-stop-words',
        action='store_true',
        help='Include common stop words in analysis'
    )
    parser.add_argument(
        '--no-ngrams',
        action='store_true',
        help='Disable n-gram analysis'
    )
    parser.add_argument(
        '--max-ngram-size',
        type=int,
        default=3,
        help='Maximum n-gram size to analyze (default: 3)'
    )
    parser.add_argument(
        '--ngram-output',
        help='Output file path for n-gram analysis (default: same directory as input file)',
        default=None
    )
    parser.add_argument(
        '--print-results',
        action='store_true',
        help='Print results to console'
    )
    parser.add_argument(
        '--analyze-failed',
        action='store_true',
        help='Also analyze failed translations and save to separate file'
    )
    parser.add_argument(
        '--failed-output',
        help='Output file path for failed translations analysis (default: same directory as input file)',
        default=None
    )
    parser.add_argument(
        '--failed-format',
        choices=['json', 'csv', 'txt'],
        default='json',
        help='Output format for failed translations analysis (default: json)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = WordAnalyzer(
        min_word_length=args.min_length,
        exclude_common_words=not args.no_stop_words,
        include_ngrams=not args.no_ngrams,
        max_ngram_size=args.max_ngram_size
    )
    
    try:
        # Analyze word frequency
        print(f"Analyzing word frequency in: {args.input_file}")
        results = analyzer.get_most_common_words(args.input_file, args.top_n)
        
        if not results:
            print("No words found matching the criteria")
            return
        
        # Determine output path
        if args.output is None:
            output_path = analyzer.get_default_output_path(args.input_file, args.format)
        else:
            output_path = args.output
        
        # Save results
        analyzer.save_results(results, output_path, args.format)
        print(f"Results saved to: {output_path}")
        
        # Print results if requested
        if args.print_results:
            print(f"\nTop {len(results)} most common words:")
            print("-" * 40)
            for i, (word, freq) in enumerate(results, 1):
                print(f"{i:3d}. {word:<20} {freq:>6}")
        
        # Print summary
        total_words = sum(freq for _, freq in results)
        print(f"\nSummary:")
        print(f"Total unique words analyzed: {len(results)}")
        print(f"Total word occurrences: {total_words}")
        
        # Analyze n-grams if enabled
        if analyzer.include_ngrams:
            print(f"\nAnalyzing n-grams...")
            ngram_results = analyzer.get_most_common_ngrams(args.input_file, args.top_n)
            
            if ngram_results:
                # Determine output path for n-grams
                if args.ngram_output is None:
                    ngram_output_path = analyzer.get_default_output_path(args.input_file, args.format).replace('_word_frequency', '_ngram_frequency')
                else:
                    ngram_output_path = args.ngram_output
                
                # Save n-gram results
                analyzer.save_ngram_results(ngram_results, ngram_output_path, args.format)
                print(f"N-gram results saved to: {ngram_output_path}")
                
                # Print n-gram summary
                total_ngrams = sum(len(results) for results in ngram_results.values())
                total_ngram_occurrences = sum(sum(freq for _, freq in results) for results in ngram_results.values())
                print(f"\nN-gram Summary:")
                print(f"Total unique n-grams analyzed: {total_ngrams}")
                print(f"Total n-gram occurrences: {total_ngram_occurrences}")
                
                # Print top n-grams for each size if requested
                if args.print_results:
                    for n, ngram_list in ngram_results.items():
                        if ngram_list:
                            print(f"\nTop {len(ngram_list)} most common {n}-grams:")
                            print("-" * 50)
                            for i, (ngram, freq) in enumerate(ngram_list, 1):
                                print(f"{i:3d}. {ngram:<30} {freq:>6}")
            else:
                print("No n-grams found matching the criteria")
        
        # Analyze failed translations if requested
        if args.analyze_failed:
            print(f"\nAnalyzing failed translations...")
            try:
                # Analyze failed translations
                failed_analysis = analyzer.analyze_failed_translations(args.input_file)
                
                # Determine output path for failed translations
                if args.failed_output is None:
                    failed_output_path = analyzer.get_failed_translations_output_path(args.input_file, args.failed_format)
                else:
                    failed_output_path = args.failed_output
                
                # Save detailed failed translations analysis
                analyzer.save_failed_translations_analysis(failed_analysis, failed_output_path, args.failed_format)
                print(f"Failed translations analysis saved to: {failed_output_path}")
                
                # Also save summary
                summary_output_path = failed_output_path.replace(f".{args.failed_format}", f"_summary.{args.failed_format}")
                failed_summary = analyzer.get_failed_translations_summary(args.input_file)
                analyzer.save_failed_translations_summary(failed_summary, summary_output_path, args.failed_format)
                print(f"Failed translations summary saved to: {summary_output_path}")
                
                # Print summary
                print(f"\nFailed Translations Summary:")
                print(f"Total concepts analyzed: {failed_analysis['total_concepts']}")
                print(f"Failed translations: {failed_analysis['total_failed']}")
                print(f"Success rate: {failed_analysis['success_rate']}%")
                
                if failed_analysis['total_failed'] > 0:
                    print(f"\nTop 10 most frequently failed terms:")
                    for i, (term, count, categories, normalized_text, avg_confidence) in enumerate(failed_summary[:10], 1):
                        print(f"{i:2d}. {term:<20} (failed {count} times, avg confidence: {avg_confidence})")
                
            except Exception as e:
                print(f"Error analyzing failed translations: {e}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
