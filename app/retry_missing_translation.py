#!/usr/bin/env python3
"""
Script to retry missing translations.

This script identifies and retries specific missing translations from a batch.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Add the app directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from translation import TranslationRequest, translate_text, translate_without_keywords
from batch_translate import _save_translation_incrementally, _get_translation_key


def find_missing_translations(translations_file: str, expected_modes: list = None):
    """
    Find missing translations in a translations file.
    
    Args:
        translations_file: Path to the translations JSON file
        expected_modes: List of expected translation modes (default: ['with_keywords', 'without_keywords'])
    
    Returns:
        Dictionary mapping file_path to missing modes
    """
    if expected_modes is None:
        expected_modes = ['with_keywords', 'without_keywords']
    
    with open(translations_file, 'r', encoding='utf-8') as f:
        translations = json.load(f)
    
    # Group by file_path
    file_modes = defaultdict(set)
    for translation in translations:
        file_path = translation.get('file_path', '')
        mode = translation.get('translation_mode', '')
        if file_path and mode:
            file_modes[file_path].add(mode)
    
    # Find missing modes
    missing_translations = {}
    for file_path, present_modes in file_modes.items():
        missing_modes = set(expected_modes) - present_modes
        if missing_modes:
            missing_translations[file_path] = missing_modes
    
    return missing_translations


def load_dataset_for_file(dataset_path: str, target_file_path: str):
    """
    Load the specific file from the dataset.
    
    Args:
        dataset_path: Path to the dataset
        target_file_path: Target file path to find
        
    Returns:
        Dictionary with the file data or None if not found
    """
    from datasets import load_from_disk
    
    dataset = load_from_disk(dataset_path)
    
    for item in dataset:
        if item.get('file_path') == target_file_path:
            return item
    
    return None


def retry_missing_translation(
    dataset_path: str,
    translations_file: str,
    target_file_path: str,
    missing_mode: str,
    source_lang: str = "de",
    target_lang: str = "en",
    model: str = "s-rat-mini",
    use_dictionary_fallback: bool = True,
    use_file_prompts: bool = False,
    use_dictionary_only: bool = False
):
    """
    Retry a specific missing translation.
    
    Args:
        dataset_path: Path to the input dataset
        translations_file: Path to the translations JSON file
        target_file_path: File path to retry
        missing_mode: Translation mode to retry ('with_keywords' or 'without_keywords')
        source_lang: Source language code
        target_lang: Target language code
        model: Model to use
        use_dictionary_fallback: Whether to use dictionary fallback
        use_file_prompts: Whether to use system prompts from files instead of simple hardcoded prompts
        use_dictionary_only: Whether to use only dictionary translations, skipping SNOMED/UMLS lookups
    """
    print(f"Retrying translation for: {target_file_path}")
    print(f"Missing mode: {missing_mode}")
    
    # Load the file data
    file_data = load_dataset_for_file(dataset_path, target_file_path)
    if not file_data:
        print(f"❌ File not found in dataset: {target_file_path}")
        return False
    
    print(f"✅ Found file in dataset")
    
    # Create base result structure
    base_result = {
        'file_path': file_data['file_path'],
        'patient_id': file_data['patient_id'],
        'study_id': file_data['study_id'],
        'partition': file_data['partition']
    }
    
    try:
        if missing_mode == 'with_keywords':
            print("🔄 Retrying with_keywords translation...")
            translation_result = translate_text(TranslationRequest(
                text=file_data['text'],
                source_lang=source_lang,
                target_lang=target_lang,
                model=model,
                use_dictionary_fallback=use_dictionary_fallback,
                use_file_prompts=use_file_prompts,
                use_dictionary_only=use_dictionary_only
            ))
            translation_result['translation_mode'] = 'with_keywords'
            translation_result['model_used'] = model
            
        elif missing_mode == 'without_keywords':
            print("🔄 Retrying without_keywords translation...")
            translation_result = translate_without_keywords(
                text=file_data['text'],
                source_lang=source_lang,
                target_lang=target_lang,
                model=model,
                use_file_prompts=use_file_prompts
            )
        else:
            print(f"❌ Unknown translation mode: {missing_mode}")
            return False
        
        # Combine with base result
        full_result = {**base_result, **translation_result}
        
        # Save the translation
        _save_translation_incrementally(full_result, translations_file)
        
        print(f"✅ Successfully retried {missing_mode} translation for {target_file_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error retrying translation: {e}")
        return False


def main():
    """Command line interface for retrying missing translations."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Retry missing translations')
    parser.add_argument('dataset_path', help='Path to the input dataset')
    parser.add_argument('translations_file', help='Path to the translations JSON file')
    parser.add_argument('--file-path', help='Specific file path to retry (if not provided, will find all missing)')
    parser.add_argument('--mode', choices=['with_keywords', 'without_keywords'], 
                       help='Specific mode to retry (if not provided, will retry all missing modes)')
    parser.add_argument('--source-lang', default='en', help='Source language code')
    parser.add_argument('--target-lang', default='de', help='Target language code')
    parser.add_argument('--model', default='s-rat-mini', help='Model to use')
    parser.add_argument('--no-dictionary-fallback', action='store_true',
                       help='Disable dictionary fallback')
    parser.add_argument('--use-file-prompts', action='store_true',
                       help='Use system prompts from files instead of simple hardcoded prompts')
    parser.add_argument('--use-dictionary-only', action='store_true',
                       help='Use only dictionary translations, skipping SNOMED/UMLS lookups')
    
    args = parser.parse_args()
    
    # Find missing translations
    missing_translations = find_missing_translations(args.translations_file)
    
    if not missing_translations:
        print("✅ No missing translations found!")
        return 0
    
    print(f"Found {len(missing_translations)} files with missing translations:")
    for file_path, missing_modes in missing_translations.items():
        print(f"  {file_path}: missing {missing_modes}")
    
    # If specific file path provided, retry only that one
    if args.file_path:
        if args.file_path not in missing_translations:
            print(f"❌ File {args.file_path} is not missing any translations")
            return 1
        
        missing_modes = missing_translations[args.file_path]
        
        # If specific mode provided, retry only that mode
        if args.mode:
            if args.mode not in missing_modes:
                print(f"❌ Mode {args.mode} is not missing for file {args.file_path}")
                return 1
            missing_modes = [args.mode]
        
        # Retry the missing translations
        success_count = 0
        for mode in missing_modes:
            if retry_missing_translation(
                args.dataset_path,
                args.translations_file,
                args.file_path,
                mode,
                args.source_lang,
                args.target_lang,
                args.model,
                not args.no_dictionary_fallback,
                args.use_file_prompts,
                args.use_dictionary_only
            ):
                success_count += 1
        
        print(f"\n✅ Successfully retried {success_count}/{len(missing_modes)} missing translations")
        
    else:
        print("\nTo retry specific translations, use --file-path and optionally --mode")
        print("Example:")
        print(f"  python {__file__} {args.dataset_path} {args.translations_file} --file-path 'mimic-cxr-reports/p14/p14908132/s51001563.txt' --mode without_keywords")
        print(f"  python {__file__} {args.dataset_path} {args.translations_file} --file-path 'mimic-cxr-reports/p14/p14908132/s51001563.txt' --mode with_keywords --use-file-prompts")
        print(f"  python {__file__} {args.dataset_path} {args.translations_file} --file-path 'mimic-cxr-reports/p14/p14908132/s51001563.txt' --mode with_keywords --use-dictionary-only")
    
    return 0


if __name__ == '__main__':
    exit(main())
