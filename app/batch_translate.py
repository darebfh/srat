"""Module for batch translation of medical reports."""

import os
from typing import Optional, List, Dict, Any, Set
from enum import Enum
import random
from datetime import datetime
from pathlib import Path
import json
from datasets import load_from_disk
from tqdm import tqdm

import prompts
from translation import TranslationRequest, translate_text, translate_without_keywords
from llm_service import llm_service
from snomed_service import SnomedTranslationService
from umls_service import UMLSService
from extractors import AzureHealthExtractor
from config import settings
from exceptions import ModelNotFoundError, ServiceConnectionError, TranslationError

class TranslationMode(Enum):
    """Enum for different translation modes."""
    WITH_KEYWORDS = "with_keywords"
    WITHOUT_KEYWORDS = "without_keywords"
    BOTH = "both"

def _get_translation_key(file_path: str, model: str, translation_mode: str) -> str:
    """
    Generate a unique key for a translation based on file path, model, and mode.
    
    Args:
        file_path: Path to the source file
        model: Model used for translation
        translation_mode: Translation mode (with_keywords/without_keywords)
        
    Returns:
        Unique key for the translation
    """
    return f"{file_path}:{model}:{translation_mode}"

def _load_existing_translations(translations_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing translations from JSON file.
    
    Args:
        translations_path: Path to the translations JSON file
        
    Returns:
        Dictionary mapping translation keys to translation data
    """
    if not os.path.exists(translations_path):
        return {}
    
    try:
        with open(translations_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        
        # Convert list to dictionary for faster lookup
        translation_dict = {}
        for translation in translations:
            key = _get_translation_key(
                translation['file_path'],
                translation['model_used'],
                translation['translation_mode']
            )
            translation_dict[key] = translation
        
        return translation_dict
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing translations from {translations_path}: {e}")
        return {}

def _save_translation_incrementally(translation: Dict[str, Any], translations_path: str) -> None:
    """
    Save a single translation to the JSON file incrementally.
    
    Args:
        translation: Translation data to save
        translations_path: Path to the translations JSON file
    """
    # Load existing translations
    existing_translations = _load_existing_translations(translations_path)
    
    # Add new translation
    key = _get_translation_key(
        translation['file_path'],
        translation['model_used'],
        translation['translation_mode']
    )
    existing_translations[key] = translation
    
    # Convert back to list and save
    translations_list = list(existing_translations.values())
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(translations_path), exist_ok=True)
    
    with open(translations_path, 'w', encoding='utf-8') as f:
        json.dump(translations_list, f, indent=2, ensure_ascii=False)

def _is_translation_exists(file_path: str, model: str, translation_mode: str, 
                         existing_translations: Dict[str, Dict[str, Any]]) -> bool:
    """
    Check if a translation already exists.
    
    Args:
        file_path: Path to the source file
        model: Model used for translation
        translation_mode: Translation mode
        existing_translations: Dictionary of existing translations
        
    Returns:
        True if translation exists, False otherwise
    """
    key = _get_translation_key(file_path, model, translation_mode)
    return key in existing_translations

def batch_translate(
    input_dataset_path: str,
    output_dataset_path: str,
    source_lang: str = "de",
    target_lang: str = "en",
    num_reports: Optional[int] = None,
    mode: TranslationMode = TranslationMode.WITH_KEYWORDS,
    models: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    force_retranslate: bool = False,
    use_dictionary_fallback: bool = True,
    use_dictionary_only: bool = False,
    use_file_prompts: bool = False
) -> None:
    """
    Perform batch translation on a dataset of medical reports.
    
    Args:
        input_dataset_path: Path to the input dataset
        output_dataset_path: Path to save the translation results
        source_lang: Source language code
        target_lang: Target language code
        num_reports: Number of reports to translate (None for all)
        mode: Translation mode (with keywords, without, or both)
        models: List of models to use (default: [settings.DEFAULT_MODEL])
        random_seed: Random seed for reproducible sampling
        force_retranslate: If True, re-translate existing translations
        use_dictionary_fallback: Whether to use dictionary fallback for failed SNOMED translations
        use_dictionary_only: Whether to use only verified dictionary translations, skip SNOMED/UMLS
        use_file_prompts: Whether to use system prompts from files instead of simple hardcoded prompts
        
    Raises:
        ModelNotFoundError: If any specified model is not available
        ServiceConnectionError: If there's an error connecting to services
        TranslationError: For other translation-related errors
    """
    # Set default model if none provided
    if models is None:
        models = [settings.DEFAULT_MODEL]
        
    # Validate services based on mode
    if use_dictionary_only:
        print("\nValidating services for dictionary-only mode...")
        
        # Only validate LLM models for dictionary-only mode
        print("Validating LLM models...")
        for model in models:
            if not llm_service.validate_model(model):
                raise ModelNotFoundError(
                    f"\nError: Model '{model}' not found. Please pull it first using:\n"
                    f"ollama pull {model}"
                )
        print("All LLM models validated successfully.")
        
        # Validate enhanced dictionary
        print("Validating enhanced dictionary...")
        try:
            from enhanced_dictionary_service import get_enhanced_dictionary
            enhanced_dict = get_enhanced_dictionary()
            if not enhanced_dict.is_available():
                raise ServiceConnectionError("Enhanced dictionary is not available")
            print("Enhanced dictionary validated successfully.")
        except Exception as e:
            raise ServiceConnectionError(f"Enhanced dictionary validation failed: {str(e)}")
        
        print("All required services for dictionary-only mode validated successfully.")
    else:
        # Validate all services for normal mode
        print("\nValidating services...")
        
        # Validate LLM models
        print("Validating LLM models...")
        for model in models:
            if not llm_service.validate_model(model):
                raise ModelNotFoundError(
                    f"\nError: Model '{model}' not found. Please pull it first using:\n"
                    f"ollama pull {model}"
                )
        print("All LLM models validated successfully.")
        
        # Validate SNOMED service
        print("Validating SNOMED service...")
        try:
            snomed_service = SnomedTranslationService()
            snomed_service.validate_service()
            print("SNOMED service validated successfully.")
        except Exception as e:
            raise ServiceConnectionError(f"SNOMED service validation failed: {str(e)}")
        
        # Validate UMLS service
        print("Validating UMLS service...")
        try:
            umls_service = UMLSService()
            umls_service.validate_service()
            print("UMLS service validated successfully.")
        except Exception as e:
            raise ServiceConnectionError(f"UMLS service validation failed: {str(e)}")
        
        # Validate Azure Text Analytics service
        print("Validating Azure Text Analytics service...")
        try:
            azure_extractor = AzureHealthExtractor(source_lang)
            azure_extractor.validate_service()
            print("Azure Text Analytics service validated successfully.")
        except Exception as e:
            raise ServiceConnectionError(f"Azure Text Analytics service validation failed: {str(e)}")
        
        print("All services validated successfully.")
    
    # Load the dataset (Arrow format for input)
    dataset = load_from_disk(input_dataset_path)
    
    # Select reports to translate
    if num_reports is not None and num_reports < len(dataset):
        if random_seed is not None:
            random.seed(random_seed)
        indices = random.sample(range(len(dataset)), num_reports)
        dataset = dataset.select(indices)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dataset_path, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'source_lang': source_lang,
        'target_lang': target_lang,
        'num_reports': len(dataset),
        'translation_mode': mode.value,
        'models': models,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat(),
        'input_dataset': input_dataset_path
    }
    
    # Save metadata
    with open(os.path.join(output_dataset_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Set up paths for incremental saving
    translations_path = os.path.join(output_dataset_path, 'translations.json')
    errors_path = os.path.join(output_dataset_path, 'errors.json')
    
    # Load existing translations and errors to resume from where we left off
    existing_translations = _load_existing_translations(translations_path)
    existing_errors = []
    if os.path.exists(errors_path):
        try:
            with open(errors_path, 'r', encoding='utf-8') as f:
                existing_errors = json.load(f)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load existing errors from {errors_path}: {e}")
    
    results = []
    errors = existing_errors.copy()
    skipped_count = 0
    
    # Process each report
    for idx in tqdm(range(len(dataset)), desc="Translating reports"):
        report = dataset[idx]
        
        # Create base result with source information
        base_result = {
            'file_path': report['file_path'],
            'patient_id': report['patient_id'],
            'study_id': report['study_id'],
            'partition': report['partition'],
            'original_text': report['text']
        }
        
        # Process with each model
        for model in models:
            try:
                if mode in [TranslationMode.WITH_KEYWORDS, TranslationMode.BOTH]:
                    # Check if translation already exists (unless force_retranslate is True)
                    if not force_retranslate and _is_translation_exists(report['file_path'], model, 'with_keywords', existing_translations):
                        skipped_count += 1
                        print(f"Skipping existing translation: {report['file_path']} with {model} (with_keywords)")
                        continue
                    
                    # Translate with keywords
                    with_keywords = translate_text(TranslationRequest(
                        text=report['text'],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        model=model,
                        use_dictionary_fallback=use_dictionary_fallback,
                        use_dictionary_only=use_dictionary_only,
                        use_file_prompts=use_file_prompts
                    ))
                    with_keywords['translation_mode'] = 'with_keywords'
                    with_keywords['model_used'] = model
                    
                    # Save translation incrementally
                    translation_result = {**base_result, **with_keywords}
                    _save_translation_incrementally(translation_result, translations_path)
                    results.append(translation_result)
                    
                    # Update existing translations for subsequent checks
                    key = _get_translation_key(report['file_path'], model, 'with_keywords')
                    existing_translations[key] = translation_result
                
                if mode in [TranslationMode.WITHOUT_KEYWORDS, TranslationMode.BOTH]:
                    # Check if translation already exists (unless force_retranslate is True)
                    if not force_retranslate and _is_translation_exists(report['file_path'], model, 'without_keywords', existing_translations):
                        skipped_count += 1
                        print(f"Skipping existing translation: {report['file_path']} with {model} (without_keywords)")
                        continue
                    
                    # Translate without keywords
                    without_keywords = translate_without_keywords(
                        text=report['text'],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        model=model,
                        use_file_prompts=use_file_prompts
                    )
                    
                    # Save translation incrementally
                    translation_result = {**base_result, **without_keywords}
                    _save_translation_incrementally(translation_result, translations_path)
                    results.append(translation_result)
                    
                    # Update existing translations for subsequent checks
                    key = _get_translation_key(report['file_path'], model, 'without_keywords')
                    existing_translations[key] = translation_result
                    
            except (ModelNotFoundError, ServiceConnectionError) as e:
                # These are fatal errors, stop processing
                print(f"\nFatal error: {str(e)}")
                return
                
            except TranslationError as e:
                # Log translation errors but continue processing
                error_info = {
                    'file_path': report['file_path'],
                    'error': str(e),
                    'patient_id': report['patient_id'],
                    'study_id': report['study_id'],
                    'model': model,
                    'error_type': e.__class__.__name__
                }
                errors.append(error_info)
                
                # Save errors incrementally
                with open(errors_path, 'w', encoding='utf-8') as f:
                    json.dump(errors, f, indent=2, ensure_ascii=False)
                
                print(f"\nError processing {report['file_path']} with model {model}: {str(e)}")
    
    # Save final errors (in case there were any new ones)
    if errors:
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\nTranslation Summary:")
    print(f"Total reports processed: {len(dataset)}")
    print(f"New translations completed: {len(results)}")
    print(f"Skipped existing translations: {skipped_count}")
    print(f"Failed translations: {len(errors)}")
    print(f"Total translations in file: {len(existing_translations)}")
    
    # Print per-model statistics
    print("\nPer-model statistics:")
    for model in models:
        model_results = [r for r in results if r['model_used'] == model]
        print(f"\n{model}:")
        print(f"  Total translations: {len(model_results)}")
        if mode == TranslationMode.BOTH:
            with_keywords = sum(1 for r in model_results if r['translation_mode'] == 'with_keywords')
            without_keywords = sum(1 for r in model_results if r['translation_mode'] == 'without_keywords')
            print(f"  With keywords: {with_keywords}")
            print(f"  Without keywords: {without_keywords}")

def main():
    """Command line interface for batch translation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch translate medical reports.')
    parser.add_argument('input_dataset', help='Path to the input dataset')
    parser.add_argument('--output-dir', default='data/translations', 
                       help='Directory to save translation results')
    parser.add_argument('--output-name', 
                       help='Custom name for output directory (instead of timestamp)')
    parser.add_argument('--source-lang', default='en', 
                       help='Source language code (default: de)')
    parser.add_argument('--target-lang', default='de', 
                       help='Target language code (default: en)')
    parser.add_argument('--num-reports', type=int, 
                       help='Number of reports to translate (default: all)')
    parser.add_argument('--mode', choices=['with_keywords', 'without_keywords', 'both'],
                       default='with_keywords', help='Translation mode')
    parser.add_argument('--models', nargs='+', default=[settings.DEFAULT_MODEL],
                       help=f'Models to use for translation (default: {settings.DEFAULT_MODEL})')
    parser.add_argument('--random-seed', type=int,
                       help='Random seed for reproducible sampling')
    parser.add_argument('--force-retranslate', action='store_true',
                       help='Force re-translation of existing translations')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from the most recent translation directory')
    parser.add_argument('--no-dictionary-fallback', action='store_true',
                       help='Disable dictionary fallback for failed SNOMED translations')
    parser.add_argument('--use-dictionary-only', action='store_true',
                       help='Use only verified dictionary translations, skip SNOMED/UMLS lookups')
    parser.add_argument('--use-file-prompts', action='store_true',
                       help='Use system prompts from files instead of simple hardcoded prompts')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output_name:
        # Use custom output name
        output_path = os.path.join(args.output_dir, args.output_name)
        print(f"Using custom output directory: {output_path}")
    elif args.resume:
        # Find the most recent translation directory
        if os.path.exists(args.output_dir):
            subdirs = [d for d in os.listdir(args.output_dir) 
                      if os.path.isdir(os.path.join(args.output_dir, d))]
            if subdirs:
                # Sort by timestamp (directory name) and get the most recent
                # Handle both timestamp directories (YYYYMMDD_HHMMSS) and custom names
                timestamp_dirs = [d for d in subdirs if len(d) == 15 and d.replace('_', '').isdigit()]
                if timestamp_dirs:
                    latest_dir = sorted(timestamp_dirs)[-1]
                else:
                    # If no timestamp directories, use the most recent by modification time
                    latest_dir = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(args.output_dir, d)))
                output_path = os.path.join(args.output_dir, latest_dir)
                print(f"Resuming from existing directory: {output_path}")
            else:
                # No existing directories, create new one
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = os.path.join(args.output_dir, timestamp)
                print(f"No existing translation directories found. Creating new directory: {output_path}")
        else:
            # Output directory doesn't exist, create new one
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(args.output_dir, timestamp)
            print(f"Output directory doesn't exist. Creating new directory: {output_path}")
    else:
        # Create a unique output directory using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(args.output_dir, timestamp)
        print(f"Creating new translation directory: {output_path}")
    
    try:
        batch_translate(
            input_dataset_path=args.input_dataset,
            output_dataset_path=output_path,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            num_reports=args.num_reports,
            mode=TranslationMode(args.mode),
            models=args.models,
            random_seed=args.random_seed,
            force_retranslate=args.force_retranslate,
            use_dictionary_fallback=not args.no_dictionary_fallback,
            use_dictionary_only=args.use_dictionary_only,
            use_file_prompts=args.use_file_prompts
        )
    except (ModelNotFoundError, ServiceConnectionError, TranslationError) as e:
        print(f"\nError: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 