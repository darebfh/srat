"""
LLM-based evaluation module for machine translation.

This module provides functionality to evaluate translation quality using LLM-based
scoring instead of COMET metrics. The LLM scores translations on a 0-100 scale.
"""

import os
import json
import glob
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

from llm_service import llm_service
from prompts import get_language_name
from exceptions import EvaluationError
from config import settings


def find_latest_translation_dataset(base_dir: str = "data/translations") -> str:
    """
    Find the latest translation dataset directory.
    
    Args:
        base_dir: Base directory to search for translation datasets
        
    Returns:
        Path to the latest translation dataset
        
    Raises:
        EvaluationError: If no translation datasets are found
    """
    if not os.path.exists(base_dir):
        raise EvaluationError(f"Translation directory not found: {base_dir}")
    
    # Find all timestamped directories
    pattern = os.path.join(base_dir, "*/translations.json")
    dataset_files = glob.glob(pattern)
    
    if not dataset_files:
        raise EvaluationError(f"No translation datasets found in {base_dir}")
    
    # Sort by directory name (timestamp) to get the latest
    latest_dataset = sorted(dataset_files)[-1]
    
    # Verify the dataset exists and is valid
    if not os.path.exists(latest_dataset):
        raise EvaluationError(f"Dataset file not found: {latest_dataset}")
    
    print(f"Found latest translation dataset: {latest_dataset}")
    return latest_dataset


def extract_score_from_response(response_text: str) -> Optional[float]:
    """
    Extract a numeric score (0-100) from LLM response.
    
    Args:
        response_text: The LLM response text
        
    Returns:
        Extracted score as float, or None if not found
    """
    # Try to find a number between 0 and 100
    # Look for patterns like "85", "85.5", "Score: 85", etc.
    patterns = [
        r'\b(\d{1,2}(?:\.\d+)?)\b',  # Any number (will filter to 0-100)
        r'[Ss]core[:\s]+(\d{1,2}(?:\.\d+)?)',
        r'(\d{1,2}(?:\.\d+)?)\s*(?:out of|/)\s*100',
        r'(\d{1,2}(?:\.\d+)?)\s*%',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            # Get the first match and convert to float
            try:
                score = float(matches[0])
                # Ensure it's in valid range
                if 0 <= score <= 100:
                    return score
            except (ValueError, IndexError):
                continue
    
    # If no pattern matched, try to find any number and validate
    numbers = re.findall(r'\b\d{1,2}(?:\.\d+)?\b', response_text)
    for num_str in numbers:
        try:
            score = float(num_str)
            if 0 <= score <= 100:
                return score
        except ValueError:
            continue
    
    return None


class LLMEvaluator:
    """LLM-based evaluation for machine translation."""
    
    def __init__(self, model: str = None, temperature: float = 0.0):
        """
        Initialize the LLM evaluator.
        
        Args:
            model: LLM model to use for evaluation (defaults to config default)
            temperature: Temperature for LLM generation (0.0 for deterministic scoring)
            
        Raises:
            EvaluationError: If model validation fails
        """
        self.model = model or settings.DEFAULT_MODEL
        self.temperature = temperature
        
        # Validate model
        if not llm_service.validate_model(self.model):
            raise EvaluationError(
                f"Model '{self.model}' not found. Please pull it first using: ollama pull {self.model}"
            )
        
        print(f"Initialized LLM evaluator with model: {self.model}")
    
    def create_evaluation_prompt(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Create the evaluation prompt for scoring a translation.
        
        Args:
            source_text: Source text in source language
            target_text: Translated text in target language
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Formatted evaluation prompt
        """
        source_lang_name = get_language_name(source_lang)
        target_lang_name = get_language_name(target_lang)
        
        prompt = f"""Score the following translation from {source_lang_name} to {target_lang_name} on a continuous scale from 0 to 100, where score of zero means
"no meaning preserved" and score of one hundred means "perfect meaning and grammar".
{source_lang_name} source: "{source_text}"
{target_lang_name} translation: "{target_text}"
Score:"""
        
        return prompt
    
    def score_translation(
        self,
        source_text: str,
        target_text: str,
        source_lang: str = "de",
        target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Score a single translation using LLM.
        
        Args:
            source_text: Source text in source language
            target_text: Translated text in target language
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Dictionary containing score and metadata
        """
        if not source_text or not target_text:
            return {
                'score': None,
                'raw_response': None,
                'error': 'Empty source or target text'
            }
        
        try:
            # Create evaluation prompt
            user_prompt = self.create_evaluation_prompt(
                source_text, target_text, source_lang, target_lang
            )
            
            # System prompt for evaluation
            system_prompt = "You are an expert translator evaluator. Provide only a numeric score from 0 to 100."
            
            # Get LLM response
            response = llm_service.generate_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=self.temperature
            )
            
            response_text = response['text']
            score = extract_score_from_response(response_text)
            
            return {
                'score': score,
                'raw_response': response_text,
                'model': self.model,
                'error': None if score is not None else 'Failed to extract score from response'
            }
            
        except Exception as e:
            return {
                'score': None,
                'raw_response': None,
                'error': str(e)
            }
    
    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        source_lang: str = "de",
        target_lang: str = "en",
        batch_size: int = 1  # LLM evaluation is sequential
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset using LLM scoring.
        
        Args:
            dataset: List of dictionaries containing translations
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Not used for LLM (kept for API compatibility)
            
        Returns:
            Dictionary containing evaluation results
        """
        if not dataset:
            raise EvaluationError("Dataset is empty")
        
        print(f"Evaluating {len(dataset)} translation pairs using LLM...")
        
        scores = []
        detailed_results = []
        errors = []
        
        # Evaluate each translation
        for i, item in enumerate(tqdm(dataset, desc="Evaluating translations")):
            source_text = item.get('original_text', '')
            target_text = item.get('translated_text', '')
            
            if not source_text or not target_text:
                errors.append({
                    'index': i,
                    'error': 'Missing source or target text',
                    'file_path': item.get('file_path', '')
                })
                continue
            
            # Score the translation
            result = self.score_translation(
                source_text=source_text,
                target_text=target_text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            score = result['score']
            
            if score is not None:
                scores.append(score)
            else:
                errors.append({
                    'index': i,
                    'error': result.get('error', 'Unknown error'),
                    'file_path': item.get('file_path', ''),
                    'raw_response': result.get('raw_response', '')
                })
            
            # Create detailed result
            detailed_results.append({
                'index': i,
                'src': source_text,
                'mt': target_text,
                'llm_score': float(score) if score is not None else None,
                'raw_response': result.get('raw_response', ''),
                'file_path': item.get('file_path', ''),
                'patient_id': item.get('patient_id', ''),
                'study_id': item.get('study_id', ''),
                'model_used': item.get('model_used', ''),
                'translation_mode': item.get('translation_mode', ''),
                'evaluation_error': result.get('error')
            })
        
        if not scores:
            raise EvaluationError("No valid scores obtained from LLM evaluation")
        
        # Calculate statistics
        scores_array = np.array(scores)
        system_score = float(np.mean(scores_array))
        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array))
        min_score = float(np.min(scores_array))
        max_score = float(np.max(scores_array))
        
        return {
            'model_name': self.model,
            'num_translations': len(dataset),
            'num_successful_scores': len(scores),
            'num_errors': len(errors),
            'system_score': system_score,
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': min_score,
            'max_score': max_score,
            'scores': [float(s) for s in scores],
            'detailed_results': detailed_results,
            'errors': errors,
            'evaluation_timestamp': datetime.now().isoformat()
        }


def evaluate_translations(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations_llm",
    model: str = None,
    source_lang: str = "de",
    target_lang: str = "en",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Evaluate translations using LLM-based scoring.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save evaluation results
        model: LLM model to use for evaluation (None to use default)
        source_lang: Source language code
        target_lang: Target language code
        temperature: Temperature for LLM generation
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Find the latest dataset if not provided
        if translation_dataset_path is None:
            translation_dataset_path = find_latest_translation_dataset()
        
        # Load the translation dataset
        print(f"Loading translation dataset from: {translation_dataset_path}")
        with open(translation_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} translations")
        
        # Initialize evaluator
        evaluator = LLMEvaluator(model=model, temperature=temperature)
        
        # Evaluate the dataset
        results = evaluator.evaluate_dataset(
            dataset,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Add evaluation scores to each translation in the original dataset
        enhanced_dataset = []
        for i, translation in enumerate(dataset):
            # Find the corresponding evaluation result
            evaluation_result = None
            for detail in results['detailed_results']:
                if (detail['index'] == i or
                    (detail['src'] == translation.get('original_text', '') and 
                     detail['mt'] == translation.get('translated_text', ''))):
                    evaluation_result = detail
                    break
            
            # Create enhanced translation with evaluation score
            enhanced_translation = translation.copy()
            if evaluation_result and evaluation_result['llm_score'] is not None:
                enhanced_translation['evaluation'] = {
                    'llm_score': evaluation_result['llm_score'],
                    'model_name': results['model_name'],
                    'evaluation_timestamp': results['evaluation_timestamp'],
                    'raw_response': evaluation_result.get('raw_response', '')
                }
            else:
                enhanced_translation['evaluation'] = {
                    'llm_score': None,
                    'model_name': results['model_name'],
                    'evaluation_timestamp': results['evaluation_timestamp'],
                    'error': evaluation_result.get('evaluation_error', 'No evaluation score available') if evaluation_result else 'No evaluation result found'
                }
            
            enhanced_dataset.append(enhanced_translation)
        
        # Save enhanced dataset with evaluation scores
        enhanced_dataset_path = os.path.join(output_path, 'translations_with_evaluation.json')
        with open(enhanced_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
        
        # Save detailed results
        detailed_results_path = os.path.join(output_path, 'detailed_results.json')
        with open(detailed_results_path, 'w', encoding='utf-8') as f:
            json.dump(results['detailed_results'], f, indent=2, ensure_ascii=False)
        
        # Save summary results
        summary_results = {k: v for k, v in results.items() if k not in ['detailed_results', 'errors']}
        summary_path = os.path.join(output_path, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        # Save errors if any
        if results.get('errors'):
            errors_path = os.path.join(output_path, 'evaluation_errors.json')
            with open(errors_path, 'w', encoding='utf-8') as f:
                json.dump(results['errors'], f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("LLM EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {results['model_name']}")
        print(f"Number of translations: {results['num_translations']}")
        print(f"Successful scores: {results['num_successful_scores']}")
        print(f"Errors: {results['num_errors']}")
        print(f"System score (mean): {results['system_score']:.4f}")
        print(f"Mean score: {results['mean_score']:.4f}")
        print(f"Standard deviation: {results['std_score']:.4f}")
        print(f"Min score: {results['min_score']:.4f}")
        print(f"Max score: {results['max_score']:.4f}")
        print(f"Enhanced dataset saved to: {enhanced_dataset_path}")
        print(f"Results saved to: {output_path}")
        print("="*50)
        
        return results
        
    except Exception as e:
        raise EvaluationError(f"LLM evaluation failed: {str(e)}")


def compare_translation_modes(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations_llm",
    model: str = None,
    source_lang: str = "de",
    target_lang: str = "en",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Compare different translation modes (with/without keywords) using LLM-based scoring.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save comparison results
        model: LLM model to use for evaluation (None to use default)
        source_lang: Source language code
        target_lang: Target language code
        temperature: Temperature for LLM generation
        
    Returns:
        Dictionary containing comparison results
    """
    try:
        # Find the latest dataset if not provided
        if translation_dataset_path is None:
            translation_dataset_path = find_latest_translation_dataset()
        
        # Load the translation dataset
        print(f"Loading translation dataset from: {translation_dataset_path}")
        with open(translation_dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Separate by translation mode
        with_keywords = [item for item in dataset if item.get('translation_mode') == 'with_keywords']
        without_keywords = [item for item in dataset if item.get('translation_mode') == 'without_keywords']
        
        print(f"Found {len(with_keywords)} translations with keywords")
        print(f"Found {len(without_keywords)} translations without keywords")
        
        # Initialize evaluator
        evaluator = LLMEvaluator(model=model, temperature=temperature)
        
        comparison_results = {
            'model_name': model or settings.DEFAULT_MODEL,
            'evaluation_timestamp': datetime.now().isoformat(),
            'comparison': {}
        }
        
        # Evaluate each mode
        if with_keywords:
            print("\nEvaluating translations with keywords...")
            with_keywords_results = evaluator.evaluate_dataset(
                with_keywords,
                source_lang=source_lang,
                target_lang=target_lang
            )
            comparison_results['comparison']['with_keywords'] = {
                'num_translations': with_keywords_results['num_translations'],
                'num_successful_scores': with_keywords_results['num_successful_scores'],
                'system_score': with_keywords_results['system_score'],
                'mean_score': with_keywords_results['mean_score'],
                'std_score': with_keywords_results['std_score']
            }
        
        if without_keywords:
            print("\nEvaluating translations without keywords...")
            without_keywords_results = evaluator.evaluate_dataset(
                without_keywords,
                source_lang=source_lang,
                target_lang=target_lang
            )
            comparison_results['comparison']['without_keywords'] = {
                'num_translations': without_keywords_results['num_translations'],
                'num_successful_scores': without_keywords_results['num_successful_scores'],
                'system_score': without_keywords_results['system_score'],
                'mean_score': without_keywords_results['mean_score'],
                'std_score': without_keywords_results['std_score']
            }
        
        # Calculate difference if both modes exist
        if with_keywords and without_keywords:
            diff = (comparison_results['comparison']['with_keywords']['mean_score'] - 
                   comparison_results['comparison']['without_keywords']['mean_score'])
            comparison_results['comparison']['difference'] = {
                'mean_score_diff': float(diff),
                'better_mode': 'with_keywords' if diff > 0 else 'without_keywords'
            }
        
        # Save results
        os.makedirs(output_path, exist_ok=True)
        comparison_path = os.path.join(output_path, 'mode_comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        # Also create enhanced dataset with evaluation scores for both modes
        enhanced_dataset = []
        
        # Collect all evaluation results from both modes
        all_evaluation_results = []
        if with_keywords:
            all_evaluation_results.extend(with_keywords_results['detailed_results'])
        if without_keywords:
            all_evaluation_results.extend(without_keywords_results['detailed_results'])
        
        for i, translation in enumerate(dataset):
            # Find the corresponding evaluation result
            evaluation_result = None
            for detail in all_evaluation_results:
                if (detail['src'] == translation.get('original_text', '') and 
                    detail['mt'] == translation.get('translated_text', '')):
                    evaluation_result = detail
                    break
            
            # Create enhanced translation with evaluation score
            enhanced_translation = translation.copy()
            if evaluation_result and evaluation_result['llm_score'] is not None:
                enhanced_translation['evaluation'] = {
                    'llm_score': evaluation_result['llm_score'],
                    'model_name': comparison_results['model_name'],
                    'evaluation_timestamp': comparison_results['evaluation_timestamp'],
                    'raw_response': evaluation_result.get('raw_response', '')
                }
            else:
                enhanced_translation['evaluation'] = {
                    'llm_score': None,
                    'model_name': comparison_results['model_name'],
                    'evaluation_timestamp': comparison_results['evaluation_timestamp'],
                    'error': evaluation_result.get('evaluation_error', 'No evaluation score available') if evaluation_result else 'No evaluation result found'
                }
            
            enhanced_dataset.append(enhanced_translation)
        
        # Save enhanced dataset with evaluation scores
        enhanced_dataset_path = os.path.join(output_path, 'translations_with_evaluation.json')
        with open(enhanced_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
        
        # Print comparison
        print("\n" + "="*50)
        print("LLM MODE COMPARISON SUMMARY")
        print("="*50)
        for mode, results in comparison_results['comparison'].items():
            if mode != 'difference':
                print(f"\n{mode.replace('_', ' ').title()}:")
                print(f"  Number of translations: {results['num_translations']}")
                print(f"  Successful scores: {results['num_successful_scores']}")
                print(f"  System score (mean): {results['system_score']:.4f}")
                print(f"  Mean score: {results['mean_score']:.4f}")
                print(f"  Standard deviation: {results['std_score']:.4f}")
        
        if 'difference' in comparison_results['comparison']:
            diff_info = comparison_results['comparison']['difference']
            print(f"\nDifference: {diff_info['mean_score_diff']:.4f}")
            print(f"Better mode: {diff_info['better_mode']}")
        
        print(f"\nComparison results saved to: {output_path}")
        print("="*50)
        
        return comparison_results
        
    except Exception as e:
        raise EvaluationError(f"LLM mode comparison failed: {str(e)}")


def run_complete_evaluation_workflow(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations_llm",
    model: str = None,
    source_lang: str = "de",
    target_lang: str = "en",
    temperature: float = 0.0,
    create_excel_comparison: bool = True,
    excel_output_path: str = "data/comparisons"
) -> Dict[str, Any]:
    """
    Run the complete evaluation workflow: evaluate translations and optionally create Excel comparison.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save evaluation results
        model: LLM model to use for evaluation (None to use default)
        source_lang: Source language code
        target_lang: Target language code
        temperature: Temperature for LLM generation
        create_excel_comparison: Whether to create Excel comparison file
        excel_output_path: Path to save Excel comparison file
        
    Returns:
        Dictionary containing evaluation results and paths to created files
    """
    try:
        # Run evaluation with mode comparison
        print("Starting LLM evaluation workflow...")
        evaluation_results = compare_translation_modes(
            translation_dataset_path=translation_dataset_path,
            output_path=output_path,
            model=model,
            source_lang=source_lang,
            target_lang=target_lang,
            temperature=temperature
        )
        
        result_paths = {
            'evaluation_summary': os.path.join(output_path, 'evaluation_summary.json'),
            'detailed_results': os.path.join(output_path, 'detailed_results.json'),
            'enhanced_dataset': os.path.join(output_path, 'translations_with_evaluation.json'),
            'mode_comparison': os.path.join(output_path, 'mode_comparison.json')
        }
        
        # Create Excel comparison if requested
        if create_excel_comparison:
            print("\nCreating Excel comparison...")
            try:
                from excel_comparison import create_excel_comparison
                # Find the enhanced dataset from the evaluation
                enhanced_dataset_path = result_paths['enhanced_dataset']
                if os.path.exists(enhanced_dataset_path):
                    excel_path = create_excel_comparison(
                        dataset_path=enhanced_dataset_path,
                        output_path=excel_output_path
                    )
                    result_paths['excel_comparison'] = excel_path
                    print(f"Excel comparison created: {excel_path}")
                else:
                    print(f"Warning: Enhanced dataset not found at {enhanced_dataset_path}")
                    result_paths['excel_comparison'] = None
            except Exception as e:
                print(f"Warning: Failed to create Excel comparison: {str(e)}")
                result_paths['excel_comparison'] = None
        
        print("\n" + "="*50)
        print("LLM EVALUATION WORKFLOW COMPLETED")
        print("="*50)
        print(f"Enhanced dataset: {result_paths['enhanced_dataset']}")
        if result_paths.get('excel_comparison'):
            print(f"Excel comparison: {result_paths['excel_comparison']}")
        print("="*50)
        
        return {
            'evaluation_results': evaluation_results,
            'result_paths': result_paths
        }
        
    except Exception as e:
        raise EvaluationError(f"Complete LLM evaluation workflow failed: {str(e)}")


def main():
    """Command line interface for LLM-based evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate translations using LLM-based scoring.')
    parser.add_argument('translation_dataset', nargs='?', default=None,
                       help='Path to the translation dataset (optional, uses latest if not provided)')
    parser.add_argument('--output-dir', default='data/evaluations_llm', 
                       help='Directory to save evaluation results')
    parser.add_argument('--model', default=None,
                       help='LLM model to use for evaluation (defaults to config default)')
    parser.add_argument('--source-lang', default='de',
                       help='Source language code (default: de)')
    parser.add_argument('--target-lang', default='en',
                       help='Target language code (default: en)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for LLM generation (default: 0.0 for deterministic)')
    parser.add_argument('--compare-modes', action='store_true',
                       help='Compare different translation modes')
    parser.add_argument('--complete-workflow', action='store_true',
                       help='Run complete evaluation workflow including Excel comparison')
    parser.add_argument('--excel-output-dir', default='data/comparisons',
                       help='Directory to save Excel comparison file')
    
    args = parser.parse_args()
    
    # Create a unique output directory using timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(args.output_dir, timestamp)
    
    try:
        if args.complete_workflow:
            run_complete_evaluation_workflow(
                translation_dataset_path=args.translation_dataset,
                output_path=output_path,
                model=args.model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                temperature=args.temperature,
                create_excel_comparison=True,
                excel_output_path=args.excel_output_dir
            )
        elif args.compare_modes:
            compare_translation_modes(
                translation_dataset_path=args.translation_dataset,
                output_path=output_path,
                model=args.model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                temperature=args.temperature
            )
        else:
            evaluate_translations(
                translation_dataset_path=args.translation_dataset,
                output_path=output_path,
                model=args.model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                temperature=args.temperature
            )
    except EvaluationError as e:
        print(f"\nError: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

