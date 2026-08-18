# Descriptive Statistics Module

This module provides comprehensive descriptive statistics for the subset of 500 medical reports used in the translation experiments.

## Features

The `DescriptiveStatistics` class provides the following analyses:

### 1. Basic Statistics
- Total number of reports, patients, and studies
- File size statistics (total, average, median, standard deviation)
- Text length statistics (characters and words)
- Distribution analysis

### 2. Partition Analysis
- Distribution of reports across different partitions (p10, p11, etc.)
- Percentage breakdown by partition
- Most and least common partitions

### 3. Patient Analysis
- Number of reports per patient
- Patients with multiple reports
- Statistical distribution of reports per patient

### 4. Text Analysis
- Character, word, sentence, and paragraph counts
- Comprehensive readability metrics using py-readability-metrics:
  - Flesch Reading Ease
  - Flesch-Kincaid Grade Level
  - Automated Readability Index (ARI)
  - SMOG Index
- Text length distributions

### 5. Medical Terminology Analysis
- Analysis of common medical terms across categories:
  - Anatomy terms (heart, lung, chest, etc.)
  - Medical procedures (surgery, biopsy, scan, etc.)
  - Medical conditions (fracture, tumor, cancer, etc.)
  - Medications and dosages
  - Measurements and units

### 6. Visualization (Optional)
- Partition distribution pie chart
- Text length distribution histogram
- Word count distribution histogram
- Medical terminology bar chart
- Patient report distribution histogram

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Generate basic statistics and print summary
pdm run descriptive-stats

# Generate full analysis with visualizations and Excel export
pdm run descriptive-stats-full
```

#### Advanced Usage
```bash
# Custom dataset path
python app/descriptive_statistics.py --dataset-path /path/to/dataset --print-summary

# Create visualizations
python app/descriptive_statistics.py --create-visualizations --viz-output-dir my_plots

# Export to different formats
python app/descriptive_statistics.py --format excel --output my_stats.xlsx
python app/descriptive_statistics.py --format csv --output my_stats.csv

# Full analysis with all features
python app/descriptive_statistics.py \
    --print-summary \
    --create-visualizations \
    --format excel \
    --output comprehensive_stats.xlsx
```

### Programmatic Usage

```python
from app.descriptive_statistics import DescriptiveStatistics

# Initialize analyzer
analyzer = DescriptiveStatistics("data/processed/raw_dataset")

# Load dataset
analyzer.load_dataset()

# Get comprehensive statistics
stats = analyzer.get_comprehensive_statistics()

# Print summary
analyzer.print_summary()

# Save results
analyzer.save_results("results.json", format="json")

# Create visualizations
analyzer.create_visualizations("plots/")
```

## Output Formats

### JSON Format
Complete statistics in structured JSON format with all analysis results.

### CSV Format
Flattened statistics suitable for spreadsheet analysis.

### Excel Format
Multi-sheet Excel workbook with:
- Summary sheet with key metrics
- Detailed statistics sheet
- Medical terminology analysis sheet

### Visualizations
High-quality PNG plots (300 DPI) including:
- `partition_distribution.png` - Pie chart of partition distribution
- `text_length_distribution.png` - Histogram of text lengths
- `word_count_distribution.png` - Histogram of word counts
- `medical_terminology_analysis.png` - Bar chart of top medical terms
- `patient_distribution.png` - Histogram of reports per patient

## Dependencies

### Required
- `datasets` - For loading Arrow format datasets
- `statistics` - Python standard library for statistical calculations
- `py-readability-metrics` - For accurate readability calculations

### Optional
- `pandas` - For advanced data manipulation and Excel export
- `matplotlib` - For creating visualizations
- `seaborn` - For enhanced plot styling
- `openpyxl` - For Excel file writing

## Example Output

```
============================================================
DESCRIPTIVE STATISTICS SUMMARY
============================================================

Dataset Overview:
  Total Reports: 500
  Unique Patients: 450
  Unique Studies: 500
  Total Size: 2.45 MB

Text Statistics:
  Average Length: 1,250 characters
  Average Words: 200 words
  Median Words: 185 words

Partition Distribution:
  p10: 50 reports (10.0%)
  p11: 52 reports (10.4%)
  p12: 48 reports (9.6%)
  ...

Patient Statistics:
  Average Reports per Patient: 1.11
  Patients with Multiple Reports: 45 (10.0%)

Medical Terminology:
  Total Medical Term Occurrences: 15,420
  Top 5 Most Common Terms:
    1. heart: 1,250 occurrences
    2. lung: 980 occurrences
    3. chest: 850 occurrences
    4. scan: 720 occurrences
    5. x-ray: 650 occurrences
============================================================
```

## Integration with Existing Workflow

This module integrates seamlessly with the existing translation workflow:

1. **Dataset Creation**: Use `pdm run convert-dataset` to create the raw dataset
2. **Translation**: Use `pdm run run-translation` to translate the 500 reports
3. **Analysis**: Use `pdm run descriptive-stats` to analyze the subset
4. **Evaluation**: Use `pdm run evaluate-translations` to evaluate translations

## Notes

- The module automatically handles the 500-report subset used in translation experiments
- All statistics are calculated on the actual dataset used for translation
- Medical terminology analysis focuses on common medical terms relevant to radiology reports
- Readability metrics use the py-readability-metrics package for accurate calculations
- Visualizations are saved as high-quality PNG files suitable for publications

## Troubleshooting

### Common Issues

1. **"datasets library not available"**
   - This usually occurs when running the script directly with `python` instead of `pdm`
   - Use `pdm run descriptive-stats` instead of `python app/descriptive_statistics.py`
   - If needed, install with: `pip install datasets`

2. **"matplotlib/seaborn not available"**
   - Install with: `pip install matplotlib seaborn`
   - Visualization features will be disabled if not available

3. **"Dataset not found"**
   - Ensure you've run `pdm run convert-dataset` first
   - Check that the dataset path is correct

4. **Excel export errors**
   - The script will automatically fall back to JSON format if Excel export fails
   - Ensure `openpyxl` is installed: `pip install openpyxl`

5. **Memory issues with large datasets**
   - The module is optimized for the 500-report subset
   - For larger datasets, consider processing in batches

### Important Notes

- **Always use `pdm run` commands** for proper dependency management
- The script defaults to analyzing the same 500-report subset used in translation experiments
- Use `--all-reports` flag to analyze the entire dataset if needed
