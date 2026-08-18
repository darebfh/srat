#!/usr/bin/env python3
"""
Dictionary converter for translations.

This script converts translation JSON files into easy-to-edit CSV files
with concept-translation pairs that can be used as a fallback dictionary.

It supports two modes:
1. Failed translations only (from failed translations summary JSON)
2. All translations (from full translations JSON with successful and failed translations)
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set


class DictionaryConverter:
    """Converts translations to a dictionary CSV file for manual verification."""
    
    def __init__(self):
        self.concept_translations = {}
        self.seen_terms = set()
    
    def load_failed_translations(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load failed translations from JSON file.
        
        Args:
            json_file_path: Path to the failed translations summary JSON file
            
        Returns:
            List of failed translation dictionaries
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                failed_translations = json.load(f)
            return failed_translations
        except FileNotFoundError:
            raise FileNotFoundError(f"Failed translations file not found: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
    
    def generate_translations(self, term: str, normalized_text: str, categories: str) -> Dict[str, str]:
        """
        Generate translations for a term based on its context.
        
        Args:
            term: Original term
            normalized_text: Normalized version of the term
            categories: Medical categories
            
        Returns:
            Dictionary with language codes as keys and translations as values
        """
        translations = {}
        
        # Common medical term translations
        medical_translations = {
            # Body structures
            'chest': {'de': 'Brustkorb', 'fr': 'thorax', 'es': 'tórax', 'it': 'torace'},
            'lung': {'de': 'Lunge', 'fr': 'poumon', 'es': 'pulmón', 'it': 'polmone'},
            'heart': {'de': 'Herz', 'fr': 'cœur', 'es': 'corazón', 'it': 'cuore'},
            'brain': {'de': 'Gehirn', 'fr': 'cerveau', 'es': 'cerebro', 'it': 'cervello'},
            'liver': {'de': 'Leber', 'fr': 'foie', 'es': 'hígado', 'it': 'fegato'},
            'kidney': {'de': 'Niere', 'fr': 'rein', 'es': 'riñón', 'it': 'rene'},
            
            # Symptoms and signs
            'pain': {'de': 'Schmerz', 'fr': 'douleur', 'es': 'dolor', 'it': 'dolore'},
            'fever': {'de': 'Fieber', 'fr': 'fièvre', 'es': 'fiebre', 'it': 'febbre'},
            'swelling': {'de': 'Schwellung', 'fr': 'gonflement', 'es': 'hinchazón', 'it': 'gonfiore'},
            'bleeding': {'de': 'Blutung', 'fr': 'saignement', 'es': 'sangrado', 'it': 'sanguinamento'},
            'nausea': {'de': 'Übelkeit', 'fr': 'nausée', 'es': 'náusea', 'it': 'nausea'},
            'vomiting': {'de': 'Erbrechen', 'fr': 'vomissement', 'es': 'vómito', 'it': 'vomito'},
            
            # Medical procedures
            'surgery': {'de': 'Operation', 'fr': 'chirurgie', 'es': 'cirugía', 'it': 'chirurgia'},
            'biopsy': {'de': 'Biopsie', 'fr': 'biopsie', 'es': 'biopsia', 'it': 'biopsia'},
            'scan': {'de': 'Scan', 'fr': 'scanner', 'es': 'escáner', 'it': 'scansione'},
            'x-ray': {'de': 'Röntgen', 'fr': 'radiographie', 'es': 'radiografía', 'it': 'radiografia'},
            'mri': {'de': 'MRT', 'fr': 'IRM', 'es': 'RMN', 'it': 'RMN'},
            'ct': {'de': 'CT', 'fr': 'scanner', 'es': 'TC', 'it': 'TC'},
            
            # Medications
            'antibiotic': {'de': 'Antibiotikum', 'fr': 'antibiotique', 'es': 'antibiótico', 'it': 'antibiotico'},
            'painkiller': {'de': 'Schmerzmittel', 'fr': 'analgésique', 'es': 'analgésico', 'it': 'antidolorifico'},
            'insulin': {'de': 'Insulin', 'fr': 'insuline', 'es': 'insulina', 'it': 'insulina'},
            
            # Conditions
            'diabetes': {'de': 'Diabetes', 'fr': 'diabète', 'es': 'diabetes', 'it': 'diabete'},
            'hypertension': {'de': 'Hypertonie', 'fr': 'hypertension', 'es': 'hipertensión', 'it': 'ipertensione'},
            'pneumonia': {'de': 'Lungenentzündung', 'fr': 'pneumonie', 'es': 'neumonía', 'it': 'polmonite'},
            'cancer': {'de': 'Krebs', 'fr': 'cancer', 'es': 'cáncer', 'it': 'cancro'},
            
            # Gender
            'man': {'de': 'Mann', 'fr': 'homme', 'es': 'hombre', 'it': 'uomo'},
            'woman': {'de': 'Frau', 'fr': 'femme', 'es': 'mujer', 'it': 'donna'},
            'male': {'de': 'männlich', 'fr': 'masculin', 'es': 'masculino', 'it': 'maschio'},
            'female': {'de': 'weiblich', 'fr': 'féminin', 'es': 'femenino', 'it': 'femmina'},
            
            # Common abbreviations
            'pa': {'de': 'Pulmonalarterie', 'fr': 'artère pulmonaire', 'es': 'arteria pulmonar', 'it': 'arteria polmonare'},
            'bp': {'de': 'Blutdruck', 'fr': 'tension artérielle', 'es': 'presión arterial', 'it': 'pressione arteriosa'},
            'hr': {'de': 'Herzfrequenz', 'fr': 'fréquence cardiaque', 'es': 'frecuencia cardíaca', 'it': 'frequenza cardiaca'},
            'temp': {'de': 'Temperatur', 'fr': 'température', 'es': 'temperatura', 'it': 'temperatura'},
        }
        
        # Try to find exact match first
        term_lower = term.lower()
        if term_lower in medical_translations:
            return medical_translations[term_lower]
        
        # Try normalized text
        normalized_lower = normalized_text.lower()
        if normalized_lower in medical_translations:
            return medical_translations[normalized_lower]
        
        # Generate basic translations based on common patterns
        if term.isupper() and len(term) <= 3:
            # Likely an abbreviation - keep as is for now
            return {'de': term, 'fr': term, 'es': term, 'it': term}
        
        # For other terms, provide placeholder translations
        return {
            'de': f'[DE_TRANSLATION_NEEDED] {term}',
            'fr': f'[FR_TRANSLATION_NEEDED] {term}',
            'es': f'[ES_TRANSLATION_NEEDED] {term}',
            'it': f'[IT_TRANSLATION_NEEDED] {term}'
        }
    
    def convert_to_dictionary_csv(self, failed_translations: List[Dict[str, Any]], output_path: str) -> None:
        """
        Convert failed translations to a dictionary CSV file.
        
        Args:
            failed_translations: List of failed translation dictionaries
            output_path: Path to save the CSV dictionary
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'term', 'normalized_text', 'categories', 'failure_count', 'avg_confidence',
                'translation_de', 'translation_fr', 'translation_es', 'translation_it',
                'notes', 'verified'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in failed_translations:
                term = item.get('term', '')
                normalized_text = item.get('normalized_text', '')
                categories = item.get('categories', '')
                failure_count = item.get('failure_count', 0)
                avg_confidence = item.get('avg_confidence', 0)
                
                # Skip if we've already processed this term (case-insensitive)
                term_key = term.lower()
                if term_key in self.seen_terms:
                    continue
                self.seen_terms.add(term_key)
                
                # Generate translations
                translations = self.generate_translations(term, normalized_text, categories)
                
                # Write row
                writer.writerow({
                    'term': term,
                    'normalized_text': normalized_text,
                    'categories': categories,
                    'failure_count': failure_count,
                    'avg_confidence': avg_confidence,
                    'translation_de': translations.get('de', ''),
                    'translation_fr': translations.get('fr', ''),
                    'translation_es': translations.get('es', ''),
                    'translation_it': translations.get('it', ''),
                    'notes': f'Failed {failure_count} times, confidence: {avg_confidence}',
                    'verified': 'FALSE'  # Mark as unverified for manual review
                })
    
    def create_dictionary_from_failed_translations(self, input_file: str, output_file: str) -> None:
        """
        Create a dictionary CSV file from failed translations.
        
        Args:
            input_file: Path to the failed translations summary JSON file
            output_file: Path to save the dictionary CSV file
        """
        print(f"Loading failed translations from: {input_file}")
        failed_translations = self.load_failed_translations(input_file)
        
        print(f"Found {len(failed_translations)} failed translation entries")
        
        print(f"Converting to dictionary CSV: {output_file}")
        self.convert_to_dictionary_csv(failed_translations, output_file)
        
        print(f"Dictionary CSV created successfully!")
        print(f"Total unique terms: {len(self.seen_terms)}")
        print(f"\nNext steps:")
        print(f"1. Review and edit the CSV file: {output_file}")
        print(f"2. Replace placeholder translations with actual translations")
        print(f"3. Mark verified entries as 'TRUE' in the 'verified' column")
        print(f"4. Use the dictionary in your translation pipeline")
    
    def load_all_translations(self, json_file_path: str) -> List[Dict[str, Any]]:
        """
        Load all translations from a translations JSON file.
        
        Args:
            json_file_path: Path to the translations JSON file
            
        Returns:
            List of all translation dictionaries with concept data
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Extract all SNOMED translation data
            all_concept_translations = []
            for translation in translations:
                if 'snomed_translations' in translation:
                    for snomed_translation in translation['snomed_translations']:
                        # Add context information from the parent translation
                        concept_data = snomed_translation.copy()
                        concept_data.update({
                            'file_path': translation.get('file_path', ''),
                            'patient_id': translation.get('patient_id', ''),
                            'study_id': translation.get('study_id', ''),
                            'source_lang': translation.get('source_lang', ''),
                            'target_lang': translation.get('target_lang', ''),
                            'model_used': translation.get('model_used', ''),
                            'translation_mode': translation.get('translation_mode', '')
                        })
                        all_concept_translations.append(concept_data)
            
            return all_concept_translations
        except FileNotFoundError:
            raise FileNotFoundError(f"Translations file not found: {json_file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file {json_file_path}: {e}")
    
    def aggregate_concept_data(self, all_translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate concept data by term, collecting statistics and translations.
        
        Args:
            all_translations: List of all concept translation data
            
        Returns:
            List of aggregated concept dictionaries
        """
        concept_aggregates = {}
        
        for item in all_translations:
            term = item.get('term', '').strip()
            if not term:
                continue
            
            term_key = term.lower()
            
            if term_key not in concept_aggregates:
                concept_aggregates[term_key] = {
                    'term': term,
                    'normalized_text': item.get('normalized_text', ''),
                    'categories': set(),
                    'success_count': 0,
                    'failure_count': 0,
                    'total_count': 0,
                    'confidence_scores': [],
                    'translations': {
                        'de': set(),
                        'fr': set(),
                        'es': set(),
                        'it': set()
                    },
                    'sources': set(),
                    'files': set()
                }
            
            aggregate = concept_aggregates[term_key]
            
            # Update categories
            category = item.get('category', '')
            if category:
                aggregate['categories'].add(category)
            
            # Update counts
            aggregate['total_count'] += 1
            success = item.get('success', False)
            translation = item.get('translation', '')
            
            if success and translation:
                aggregate['success_count'] += 1
                # Store successful translation
                target_lang = item.get('target_lang', '')
                if target_lang in aggregate['translations']:
                    aggregate['translations'][target_lang].add(translation)
            else:
                aggregate['failure_count'] += 1
            
            # Update confidence scores
            confidence = item.get('confidence_score', 0)
            if confidence > 0:
                aggregate['confidence_scores'].append(confidence)
            
            # Update sources
            snomed_lookup = item.get('snomed_lookup', {})
            source = snomed_lookup.get('source', 'none')
            aggregate['sources'].add(source)
            
            # Update files
            file_path = item.get('file_path', '')
            if file_path:
                aggregate['files'].add(file_path)
        
        # Convert sets to strings and calculate averages
        aggregated_list = []
        for term_key, aggregate in concept_aggregates.items():
            avg_confidence = sum(aggregate['confidence_scores']) / len(aggregate['confidence_scores']) if aggregate['confidence_scores'] else 0
            
            # Get the most common translation for each language
            best_translations = {}
            for lang, translations in aggregate['translations'].items():
                if translations:
                    # Use the most common translation (for now, just take the first one)
                    best_translations[lang] = list(translations)[0]
                else:
                    best_translations[lang] = ''
            
            aggregated_list.append({
                'term': aggregate['term'],
                'normalized_text': aggregate['normalized_text'],
                'categories': ', '.join(sorted(filter(None, aggregate['categories']))),
                'success_count': aggregate['success_count'],
                'failure_count': aggregate['failure_count'],
                'total_count': aggregate['total_count'],
                'avg_confidence': round(avg_confidence, 3),
                'translations': best_translations,
                'sources': ', '.join(sorted(filter(None, aggregate['sources']))),
                'file_count': len(aggregate['files'])
            })
        
        return aggregated_list
    
    def convert_all_translations_to_dictionary_csv(self, all_translations: List[Dict[str, Any]], output_path: str) -> None:
        """
        Convert all translations to a dictionary CSV file.
        
        Args:
            all_translations: List of all translation dictionaries
            output_path: Path to save the CSV dictionary
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'term', 'normalized_text', 'categories', 'success_count', 'failure_count', 
                'total_count', 'avg_confidence', 'translation_de', 'translation_fr', 
                'translation_es', 'translation_it', 'sources', 'file_count', 'notes', 'verified'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in all_translations:
                term = item.get('term', '')
                normalized_text = item.get('normalized_text', '')
                categories = item.get('categories', '')
                success_count = item.get('success_count', 0)
                failure_count = item.get('failure_count', 0)
                total_count = item.get('total_count', 0)
                avg_confidence = item.get('avg_confidence', 0)
                translations = item.get('translations', {})
                sources = item.get('sources', '')
                file_count = item.get('file_count', 0)
                
                # Skip if we've already processed this term (case-insensitive)
                term_key = term.lower()
                if term_key in self.seen_terms:
                    continue
                self.seen_terms.add(term_key)
                
                # Create notes
                notes_parts = []
                if success_count > 0:
                    notes_parts.append(f"Success: {success_count}/{total_count}")
                if failure_count > 0:
                    notes_parts.append(f"Failed: {failure_count}/{total_count}")
                if sources:
                    notes_parts.append(f"Sources: {sources}")
                if file_count > 0:
                    notes_parts.append(f"Files: {file_count}")
                
                notes = "; ".join(notes_parts)
                
                # Write row
                writer.writerow({
                    'term': term,
                    'normalized_text': normalized_text,
                    'categories': categories,
                    'success_count': success_count,
                    'failure_count': failure_count,
                    'total_count': total_count,
                    'avg_confidence': avg_confidence,
                    'translation_de': translations.get('de', ''),
                    'translation_fr': translations.get('fr', ''),
                    'translation_es': translations.get('es', ''),
                    'translation_it': translations.get('it', ''),
                    'sources': sources,
                    'file_count': file_count,
                    'notes': notes,
                    'verified': 'FALSE'  # Mark as unverified for manual review
                })
    
    def create_dictionary_from_all_translations(self, input_file: str, output_file: str) -> None:
        """
        Create a dictionary CSV file from all translations (successful and failed).
        
        Args:
            input_file: Path to the translations JSON file
            output_file: Path to save the dictionary CSV file
        """
        print(f"Loading all translations from: {input_file}")
        all_translations = self.load_all_translations(input_file)
        
        print(f"Found {len(all_translations)} concept translation entries")
        
        print("Aggregating concept data...")
        aggregated_concepts = self.aggregate_concept_data(all_translations)
        
        print(f"Found {len(aggregated_concepts)} unique concepts")
        
        print(f"Converting to dictionary CSV: {output_file}")
        self.convert_all_translations_to_dictionary_csv(aggregated_concepts, output_file)
        
        print(f"Dictionary CSV created successfully!")
        print(f"Total unique terms: {len(self.seen_terms)}")
        print(f"\nNext steps:")
        print(f"1. Review and edit the CSV file: {output_file}")
        print(f"2. Verify translations and add missing ones")
        print(f"3. Mark verified entries as 'TRUE' in the 'verified' column")
        print(f"4. Use the dictionary in your translation pipeline")


def main():
    """Command line interface for dictionary converter."""
    parser = argparse.ArgumentParser(
        description="Convert translations to a dictionary CSV file"
    )
    parser.add_argument(
        'input_file',
        help='Path to the translations JSON file or failed translations summary JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output CSV file path (default: assets/dictionaries/concept_dictionary.csv)',
        default='assets/dictionaries/concept_dictionary.csv'
    )
    parser.add_argument(
        '--all-translations',
        action='store_true',
        help='Create dictionary from all translations (successful and failed) instead of just failed translations'
    )
    
    args = parser.parse_args()
    
    converter = DictionaryConverter()
    
    try:
        if args.all_translations:
            converter.create_dictionary_from_all_translations(args.input_file, args.output)
        else:
            converter.create_dictionary_from_failed_translations(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
