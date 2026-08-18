"""Script to create a dataset from text files without translation."""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datasets import Dataset, Features, Value
from tqdm import tqdm

# Define the dataset features/schema
FEATURES = Features({
    'file_path': Value('string'),
    'text': Value('string'),
    'processed': Value('bool'),
    'file_size': Value('int64'),
    'directory': Value('string'),
    'patient_id': Value('string'),
    'study_id': Value('string'),
    'partition': Value('string')
})

def extract_ids_from_path(file_path: str) -> Dict[str, str]:
    """
    Extract patient and study IDs from the file path.
    Expected format: .../pXX/pXXXXXXXXX/sXXXXX.txt
    where the second p-folder is the patient ID
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with patient_id and study_id
    """
    parts = Path(file_path).parts
    
    # Find all parts that start with 'p'
    p_parts = [p for p in parts if p.startswith('p')]
    # Use the last 'p' part as the patient ID (second-level folder)
    patient_id = p_parts[-1] if len(p_parts) >= 2 else ''
    
    # Get study ID from filename
    study_id = Path(file_path).stem  # Get filename without extension
    
    return {
        'patient_id': patient_id,
        'study_id': study_id,
        'partition': p_parts[0] if p_parts else ''  # Store the first p-folder as partition
    }

def process_file(file_path: str, base_dir: str) -> Dict[str, Any]:
    """
    Process a single text file and return its contents.
    
    Args:
        file_path: Path to the text file
        base_dir: Base directory for making relative paths
        
    Returns:
        Dictionary containing file path and text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        # Get relative path from base directory
        rel_path = os.path.relpath(file_path, base_dir)
        
        # Extract IDs from path
        ids = extract_ids_from_path(file_path)
        
        return {
            'file_path': rel_path,
            'text': text,
            'processed': False,  # Flag to track which entries have been translated
            'file_size': os.path.getsize(file_path),
            'directory': os.path.dirname(rel_path),
            'patient_id': ids['patient_id'],
            'study_id': ids['study_id'],
            'partition': ids['partition']
        }
    except UnicodeDecodeError as e:
        print(f"Error reading {file_path}: Not a valid UTF-8 text file")
        return None
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def find_text_files(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Recursively find all text files in a directory.
    
    Args:
        directory: Root directory to search in
        
    Returns:
        Tuple of (list of file paths, dict of file counts by extension)
    """
    text_files = []
    extension_counts = {}
    dir_counts = {}
    
    # Convert to absolute path
    abs_directory = os.path.abspath(directory)
    print(f"\nSearching for files in: {abs_directory}")
    
    for root, dirs, files in os.walk(abs_directory):
        # Count files in this directory
        rel_root = os.path.relpath(root, abs_directory)
        if files:  # Only print directories that have files
            dir_counts[rel_root] = len(files)
            print(f"Scanning directory: {rel_root} ({len(files)} files)")
            
        for file in files:
            # Count all file extensions
            ext = os.path.splitext(file)[1].lower()
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Collect text files
            if file.endswith(('.txt', '.text')):  # Added .text as alternative
                full_path = os.path.join(root, file)
                text_files.append(full_path)
                print(f"Found text file: {os.path.relpath(full_path, abs_directory)}")
    
    # Print directory structure summary
    print("\nDirectory structure:")
    for dir_path, count in sorted(dir_counts.items()):
        depth = dir_path.count(os.sep)
        indent = "  " * depth
        print(f"{indent}{dir_path}: {count} files")
                
    return text_files, extension_counts

def main():
    # Get input directory from command line or use default
    import argparse
    parser = argparse.ArgumentParser(description='Create a dataset from text files.')
    parser.add_argument('input_dir', help='Directory containing text files to process')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory for the dataset')
    args = parser.parse_args()
    
    # Convert input_dir to absolute path
    input_dir = os.path.abspath(args.input_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all text files
    text_files, extension_counts = find_text_files(input_dir)
    
    # Print file extension statistics
    print("\nFile extensions found:")
    for ext, count in sorted(extension_counts.items()):
        print(f"  {ext or '(no extension)'}: {count} files")
    
    if not text_files:
        print("\nError: No .txt or .text files found in the input directory or its subdirectories")
        print(f"Input directory: {input_dir}")
        print("Make sure this is the correct path containing the p*/p*/s*.txt files")
        return
        
    print(f"\nFound {len(text_files)} text files to process")
    
    # Process each file
    results = []
    errors = []
    for file_path in tqdm(text_files, desc="Processing files"):
        result = process_file(file_path, input_dir)
        if result:
            results.append(result)
        else:
            errors.append(file_path)
    
    # Check if we have any results
    if not results:
        print("Error: No files were successfully processed")
        if errors:
            print("\nFiles that failed to process:")
            for file_path in errors:
                print(f"  {file_path}")
        return
    
    # Create dataset with explicit schema
    dataset = Dataset.from_list(results, features=FEATURES)
    
    # Save dataset
    dataset_path = os.path.join(args.output_dir, 'raw_dataset')
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to: {dataset_path}")
    
    # Print some statistics
    print("\nDataset statistics:")
    print(f"Total examples: {len(dataset)}")
    print(f"Total size: {sum(x['file_size'] for x in results) / (1024*1024):.2f} MB")
    
    # Print partition statistics
    print("\nFile distribution by partition:")
    partition_counts = {}
    for r in results:
        partition_counts[r['partition']] = partition_counts.get(r['partition'], 0) + 1
    for partition, count in sorted(partition_counts.items()):
        print(f"  {partition}: {count} files")
        
    # Print patient statistics
    print("\nFile distribution by patient (showing first 10):")
    patient_counts = {}
    for r in results:
        patient_counts[r['patient_id']] = patient_counts.get(r['patient_id'], 0) + 1
    for patient_id, count in sorted(list(patient_counts.items())[:10]):
        print(f"  {patient_id}: {count} files")
    print(f"  ... and {len(patient_counts) - 10} more patients")
    
    # Print error summary if any
    if errors:
        print(f"\nWarning: {len(errors)} files failed to process")
        print("See above for details")
    
if __name__ == "__main__":
    main() 