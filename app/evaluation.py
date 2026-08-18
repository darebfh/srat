"""
Evaluation module for machine translation using COMET metrics.

This module provides functionality to evaluate translation quality using COMET
(Crosslingual Optimized Metric for Evaluation of Translation) models.
"""

import os
import json
import glob
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import json
from tqdm import tqdm

# Suppress multiprocessing resource tracker warnings from COMET
# This is a known issue with COMET's internal multiprocessing usage
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: COMET not available. Please install with: pip install unbabel-comet")

from exceptions import EvaluationError

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

class COMETEvaluator:
    """COMET-based evaluation for machine translation using reference-free models."""
    
    def __init__(self, model_name: str = "Unbabel/wmt22-cometkiwi-da", gpus: int = 0):
        """
        Initialize the COMET evaluator with a reference-free model.
        
        Args:
            model_name: Name of the COMET model to use (default: reference-free cometkiwi model)
            gpus: Number of GPUs to use (0 for CPU)
            
        Raises:
            EvaluationError: If COMET is not available or model loading fails
        """
        if not COMET_AVAILABLE:
            raise EvaluationError("COMET is not available. Please install with: pip install unbabel-comet")
        
        self.model_name = model_name
        self.gpus = gpus
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the COMET model."""
        try:
            print(f"Downloading COMET model: {self.model_name}")
            model_path = download_model(self.model_name)
            print(f"Loading model from: {model_path}")
            self.model = load_from_checkpoint(model_path)
            print("Model loaded successfully")
        except Exception as e:
            raise EvaluationError(f"Failed to load COMET model: {str(e)}")
    
    def prepare_data_for_comet(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Prepare dataset for COMET evaluation.
        
        Args:
            dataset: List of dictionaries containing translations
            
        Returns:
            List of dictionaries with 'src' and 'mt' keys for COMET
        """
        comet_data = []
        
        for item in dataset:
            # Extract source text and translated text
            src_text = item.get('original_text', '')
            mt_text = item.get('translated_text', '')
            
            if src_text and mt_text:
                comet_data.append({
                    'src': src_text,
                    'mt': mt_text
                })
        
        return comet_data
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate a dataset using COMET.
        
        Args:
            dataset: List of dictionaries containing translations
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            raise EvaluationError("Model not loaded")
        
        # Prepare data for COMET
        comet_data = self.prepare_data_for_comet(dataset)
        
        if not comet_data:
            raise EvaluationError("No valid translation pairs found in dataset")
        
        print(f"Evaluating {len(comet_data)} translation pairs...")
        
        try:
            # Run COMET evaluation
            model_output = self.model.predict(
                comet_data, 
                batch_size=batch_size, 
                gpus=self.gpus,
                num_workers=1  # Use single worker to avoid multiprocessing context error
            )
            
            # Extract scores
            scores = model_output.scores
            system_score = model_output.system_score
            
            # Calculate additional statistics
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # Create detailed results
            detailed_results = []
            for i, (item, score) in enumerate(zip(comet_data, scores)):
                detailed_results.append({
                    'index': i,
                    'src': item['src'],
                    'mt': item['mt'],
                    'comet_score': float(score),
                    'file_path': dataset[i].get('file_path', ''),
                    'patient_id': dataset[i].get('patient_id', ''),
                    'study_id': dataset[i].get('study_id', ''),
                    'model_used': dataset[i].get('model_used', ''),
                    'translation_mode': dataset[i].get('translation_mode', '')
                })
            
            return {
                'model_name': self.model_name,
                'num_translations': len(comet_data),
                'system_score': float(system_score),
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'min_score': float(min_score),
                'max_score': float(max_score),
                'scores': [float(s) for s in scores],
                'detailed_results': detailed_results,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise EvaluationError(f"Failed to evaluate dataset: {str(e)}")

def evaluate_translations(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations",
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    gpus: int = 0,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Evaluate translations using COMET reference-free model.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save evaluation results
        model_name: COMET model to use (default: reference-free cometkiwi model)
        gpus: Number of GPUs to use
        batch_size: Batch size for evaluation
        
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
        evaluator = COMETEvaluator(model_name=model_name, gpus=gpus)
        
        # Evaluate the dataset
        results = evaluator.evaluate_dataset(dataset, batch_size=batch_size)
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Add evaluation scores to each translation in the original dataset
        enhanced_dataset = []
        for i, translation in enumerate(dataset):
            # Find the corresponding evaluation result
            evaluation_result = None
            for detail in results['detailed_results']:
                if (detail['src'] == translation.get('original_text', '') and 
                    detail['mt'] == translation.get('translated_text', '')):
                    evaluation_result = detail
                    break
            
            # Create enhanced translation with evaluation score
            enhanced_translation = translation.copy()
            if evaluation_result:
                enhanced_translation['evaluation'] = {
                    'comet_score': evaluation_result['comet_score'],
                    'model_name': results['model_name'],
                    'evaluation_timestamp': results['evaluation_timestamp']
                }
            else:
                # Fallback: use index-based matching if exact text matching fails
                if i < len(results['scores']):
                    enhanced_translation['evaluation'] = {
                        'comet_score': results['scores'][i],
                        'model_name': results['model_name'],
                        'evaluation_timestamp': results['evaluation_timestamp']
                    }
                else:
                    enhanced_translation['evaluation'] = {
                        'comet_score': None,
                        'model_name': results['model_name'],
                        'evaluation_timestamp': results['evaluation_timestamp'],
                        'error': 'No evaluation score available'
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
        summary_results = {k: v for k, v in results.items() if k != 'detailed_results'}
        summary_path = os.path.join(output_path, 'evaluation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {results['model_name']}")
        print(f"Number of translations: {results['num_translations']}")
        print(f"System score: {results['system_score']:.4f}")
        print(f"Mean score: {results['mean_score']:.4f}")
        print(f"Standard deviation: {results['std_score']:.4f}")
        print(f"Min score: {results['min_score']:.4f}")
        print(f"Max score: {results['max_score']:.4f}")
        print(f"Enhanced dataset saved to: {enhanced_dataset_path}")
        print(f"Results saved to: {output_path}")
        print("="*50)
        
        return results
        
    except Exception as e:
        raise EvaluationError(f"Evaluation failed: {str(e)}")

def compare_translation_modes(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations",
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    gpus: int = 1,
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Compare different translation modes (with/without keywords) using COMET reference-free model.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save comparison results
        model_name: COMET model to use (default: reference-free cometkiwi model)
        gpus: Number of GPUs to use
        batch_size: Batch size for evaluation
        
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
        evaluator = COMETEvaluator(model_name=model_name, gpus=gpus)
        
        comparison_results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'comparison': {}
        }
        
        # Evaluate each mode
        if with_keywords:
            print("\nEvaluating translations with keywords...")
            with_keywords_results = evaluator.evaluate_dataset(with_keywords, batch_size)
            comparison_results['comparison']['with_keywords'] = {
                'num_translations': with_keywords_results['num_translations'],
                'system_score': with_keywords_results['system_score'],
                'mean_score': with_keywords_results['mean_score'],
                'std_score': with_keywords_results['std_score']
            }
        
        if without_keywords:
            print("\nEvaluating translations without keywords...")
            without_keywords_results = evaluator.evaluate_dataset(without_keywords, batch_size)
            comparison_results['comparison']['without_keywords'] = {
                'num_translations': without_keywords_results['num_translations'],
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
            if evaluation_result:
                enhanced_translation['evaluation'] = {
                    'comet_score': evaluation_result['comet_score'],
                    'model_name': model_name,
                    'evaluation_timestamp': comparison_results['evaluation_timestamp']
                }
            else:
                enhanced_translation['evaluation'] = {
                    'comet_score': None,
                    'model_name': model_name,
                    'evaluation_timestamp': comparison_results['evaluation_timestamp'],
                    'error': 'No evaluation score available'
                }
            
            enhanced_dataset.append(enhanced_translation)
        
        # Save enhanced dataset with evaluation scores
        enhanced_dataset_path = os.path.join(output_path, 'translations_with_evaluation.json')
        with open(enhanced_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
        
        # Print comparison
        print("\n" + "="*50)
        print("MODE COMPARISON SUMMARY")
        print("="*50)
        for mode, results in comparison_results['comparison'].items():
            if mode != 'difference':
                print(f"\n{mode.replace('_', ' ').title()}:")
                print(f"  Number of translations: {results['num_translations']}")
                print(f"  System score: {results['system_score']:.4f}")
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
        raise EvaluationError(f"Mode comparison failed: {str(e)}")

def main():
    """Command line interface for evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate translations using COMET.')
    parser.add_argument('translation_dataset', nargs='?', default=None,
                       help='Path to the translation dataset (optional, uses latest if not provided)')
    parser.add_argument('--output-dir', default='data/evaluations', 
                       help='Directory to save evaluation results')
    parser.add_argument('--model', default='Unbabel/wmt23-cometkiwi-da-xxl',
                       help='COMET model to use (reference-free)')
    parser.add_argument('--gpus', type=int, default=0,
                       help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for evaluation')
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
                model_name=args.model,
                gpus=args.gpus,
                batch_size=args.batch_size,
                create_excel_comparison=True,
                excel_output_path=args.excel_output_dir
            )
        elif args.compare_modes:
            compare_translation_modes(
                translation_dataset_path=args.translation_dataset,
                output_path=output_path,
                model_name=args.model,
                gpus=args.gpus,
                batch_size=args.batch_size
            )
        else:
            evaluate_translations(
                translation_dataset_path=args.translation_dataset,
                output_path=output_path,
                model_name=args.model,
                gpus=args.gpus,
                batch_size=args.batch_size
            )
    except EvaluationError as e:
        print(f"\nError: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0

def run_complete_evaluation_workflow(
    translation_dataset_path: Optional[str] = None,
    output_path: str = "data/evaluations",
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
    gpus: int = 0,
    batch_size: int = 8,
    create_excel_comparison: bool = True,
    excel_output_path: str = "data/comparisons"
) -> Dict[str, Any]:
    """
    Run the complete evaluation workflow: evaluate translations and optionally create Excel comparison.
    
    Args:
        translation_dataset_path: Path to the translation dataset (None to use latest)
        output_path: Path to save evaluation results
        model_name: COMET model to use
        gpus: Number of GPUs to use
        batch_size: Batch size for evaluation
        create_excel_comparison: Whether to create Excel comparison file
        excel_output_path: Path to save Excel comparison file
        
    Returns:
        Dictionary containing evaluation results and paths to created files
    """
    try:
        # Run evaluation
        print("Starting evaluation workflow...")
        evaluation_results = evaluate_translations(
            translation_dataset_path=translation_dataset_path,
            output_path=output_path,
            model_name=model_name,
            gpus=gpus,
            batch_size=batch_size
        )
        
        result_paths = {
            'evaluation_summary': os.path.join(output_path, 'evaluation_summary.json'),
            'detailed_results': os.path.join(output_path, 'detailed_results.json'),
            'enhanced_dataset': os.path.join(output_path, 'translations_with_evaluation.json')
        }
        
        # Create Excel comparison if requested
        if create_excel_comparison:
            print("\nCreating Excel comparison...")
            try:
                from excel_comparison import create_excel_comparison
                excel_path = create_excel_comparison(
                    dataset_path=result_paths['enhanced_dataset'],
                    output_path=excel_output_path
                )
                result_paths['excel_comparison'] = excel_path
                print(f"Excel comparison created: {excel_path}")
            except Exception as e:
                print(f"Warning: Failed to create Excel comparison: {str(e)}")
                result_paths['excel_comparison'] = None
        
        print("\n" + "="*50)
        print("EVALUATION WORKFLOW COMPLETED")
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
        raise EvaluationError(f"Complete evaluation workflow failed: {str(e)}")

if __name__ == "__main__":
    exit(main())
