<p align="center">
  <img src="assets/logo.png" alt="RAT logo" width="300">
</p>

# RAT - Retrieval augmented translation for medical texts

A sophisticated translation tool that uses Large Language Models to provide high-quality translations with special attention to important keywords and context. The tool includes prompt management and observability through Langfuse.

## Set-up 

1. Set up a local container instance of Azure Text Analytics for Health (paid): https://learn.microsoft.com/en-us/azure/ai-services/language-service/text-analytics-for-health/how-to/use-containers?tabs=language

Important: "When you use the Text Analytics for health container, the data contained in your API requests and responses is not visible to Microsoft, and is not used for training the model applied to your data."

2. Install dependencies:
```bash
pdm install
```

3. Copy the `.env.template` file, rename to `.env` and populate with your API keys. 

4. Start the Azure Text Analytics container (currently manually by copying the command to the cli and replacing the env variables):
```bash
pdm run start-azure
```

5. Obtain MIMIC-XCR dataset and put it into a data-folder at the project root (e.g., data/mimic-xcr-reports/p10/...). 
Convert it to a Hugginface dataset by runnning `pdm convert-dataset`
For details regarding the dataset, refer to its description on [Physionet](https://physionet.org/content/mimic-cxr/2.1.0/).
This should output the following result and generate new files in data/processed/raw_dataset/: 

``` 
Dataset statistics:
Total examples: 227835
Total size: 144.66 MB

File distribution by partition:
  p10: 22197 files
  p11: 23358 files
  p12: 22428 files
  p13: 22945 files
  p14: 22589 files
  p15: 23713 files
  p16: 22151 files
  p17: 22695 files
  p18: 22929 files
  p19: 22830 files

File distribution by patient (showing first 10):
  p19007931: 1 files
  p19254535: 5 files
  p19432472: 2 files
  p19437332: 2 files
  p19561401: 1 files
  p19802029: 1 files
  p19849311: 1 files
  p19881395: 1 files
  p19891107: 12 files
  p19932508: 1 files
  ... and 65369 more patients
```

6. Obtain and run snowstorm, an open-source terminology server. 

- Clone the repository: https://github.com/IHTSDO/snowstorm/tree/master?tab=readme-ov-file 
- Start the local server with docker compose: 
- Note that snowstorm needs to be manually with SNOMED-CT release files in the languages of interest. Depending on your country, you need to obtain a license for using SNOMED-CT and being able to download the latest release files.
- Populate snowstorm with the international edition and all national extensions, depending on which languages you want to translate to/from. More details here: https://github.com/IHTSDO/snowstorm/blob/master/docs/loading-snomed.md 
 - Hint: use SwaggerUI instead of the curl request! 

 7. Install and run ollama locally on port 11434. 


## Features

- Automatic extraction of medical terms with SNOMED CT codes
- Context-aware medical term translation
- Enhanced translation quality by incorporating standardized medical terminology
- Retry mechanism for API calls
- Support for any language pair supported by the LLM
- Prompt management and observability with Langfuse
- Detailed tracing of translation steps and performance metrics
- **Batch Translation with Incremental Saving**: Process large datasets with automatic resume capability
- **Resume Functionality**: Continue interrupted translations without losing progress
- **Translation Mode Options**: Choose between keyword-enhanced or direct translation methods

## Usage

### Quick Start

The project provides several convenient PDM scripts for different tasks:

```bash
# Convert raw text files to dataset
pdm run convert-dataset

# Run batch translation on large datasets
pdm run batch-translate data/processed/raw_dataset --output-dir data/translations

# Run translation evaluation (with/without keywords)
pdm run run-translation

# Evaluate translation quality using COMET metrics
pdm run evaluate-translations

# Display results in Streamlit interface
pdm run display-results
```

### Translation Pipeline

1. **Dataset Creation**: Convert raw medical text files to a structured dataset
   ```bash
   pdm run convert-dataset
   ```

2. **Batch Translation**: Process large datasets with incremental saving and resume capability
   ```bash
   pdm run batch-translate data/processed/raw_dataset --output-dir data/translations
   ```

3. **Translation**: Generate translations using different models and methods (for smaller datasets)
   ```bash
   pdm run run-translation
   ```

4. **Evaluation**: Assess translation quality using COMET metrics
   ```bash
   pdm run evaluate-translations
   ```

### Evaluation Features

The evaluation module provides comprehensive translation quality assessment:

- **Automatic Dataset Detection**: Finds the latest translation dataset automatically
- **COMET Metrics**: Uses state-of-the-art COMET-KIWI models for reference-free evaluation (no reference translations required)
- **Mode Comparison**: Compares translation quality with/without keyword enhancement
- **Detailed Analysis**: Provides per-translation scores and system-level metrics

#### PDM Scripts

The evaluation module is integrated with PDM scripts for easy usage:

```bash
# Basic evaluation (uses latest dataset, compares modes)
pdm run evaluate-translations

# For custom options, you can still use the Python script directly:
python app/evaluation.py --model Unbabel/XCOMET-XL --gpus 1
```

#### Python API Usage

For programmatic access, you can use the evaluation functions directly:

```python
from evaluation import evaluate_translations, compare_translation_modes

# Evaluate latest dataset
results = evaluate_translations()

# Compare translation modes
comparison = compare_translation_modes()

# Custom evaluation with specific parameters
results = evaluate_translations(
    translation_dataset_path='data/translations/20241201_120000/translations',
    output_path='data/evaluations',
    model_name='Unbabel/wmt22-cometkiwi-da',
    gpus=0,
    batch_size=8
)
```

### Evaluation Output

The evaluation module generates comprehensive results:

#### Summary Statistics
- **System Score**: Overall translation quality (reference-free evaluation)
- **Mean Score**: Average quality across all translations
- **Standard Deviation**: Consistency of translation quality
- **Min/Max Scores**: Range of translation quality

#### Detailed Results
- **Per-translation scores**: Individual COMET scores for each translation
- **Metadata**: File paths, patient IDs, model used, translation mode
- **Mode comparison**: Side-by-side comparison of different approaches

#### Output Files
- `evaluation_summary.json`: High-level statistics
- `detailed_results.json`: Per-translation scores and metadata
- `mode_comparison.json`: Comparison between translation modes

### Batch Translation

The batch translation feature allows you to process large datasets efficiently with built-in resume capability and incremental saving.

#### Key Features

- **Incremental Saving**: Each successful translation is saved immediately to prevent data loss
- **Resume Capability**: Automatically skips existing translations and continues from where it left off
- **Multiple Translation Modes**: Support for keyword-enhanced and direct translation methods
- **Error Recovery**: Failed translations are logged separately, allowing the process to continue
- **Progress Tracking**: Detailed statistics showing completed, skipped, and failed translations

#### Command Line Usage

```bash
# Basic batch translation
pdm run batch-translate data/processed/raw_dataset --output-dir data/translations

# With specific parameters
pdm run batch-translate data/processed/raw_dataset \
    --output-dir data/translations \
    --source-lang de \
    --target-lang en \
    --num-reports 100 \
    --mode both \
    --models llama3.1:8b \
    --random-seed 42

# Force re-translation of existing translations
pdm run batch-translate data/processed/raw_dataset \
    --output-dir data/translations \
    --force-retranslate

# Resume from the most recent translation directory
pdm run batch-translate data/processed/raw_dataset \
    --output-dir data/translations \
    --resume

# Use a custom output directory name (easier to resume)
pdm run batch-translate data/processed/raw_dataset \
    --output-dir data/translations \
    --output-name my_translation_batch

# Resume from custom directory
pdm run batch-translate data/processed/raw_dataset \
    --output-dir data/translations \
    --output-name my_translation_batch

# Use the predefined run-translation script (150 reports, both modes, gemma3:12b)
pdm run run-translation
```

#### Translation Modes

- `with_keywords`: Uses medical term extraction and context-aware translation (default)
- `without_keywords`: Direct translation without keyword enhancement
- `both`: Generates both types of translations for comparison

#### Output Structure

Each batch translation creates a timestamped directory containing:

```
data/translations/20241201_120000/
├── metadata.json          # Translation parameters and metadata
├── translations.json      # All successful translations
└── errors.json           # Failed translations with error details
```

#### Resume Functionality

If a batch translation is interrupted, you have several options to resume:

##### Option 1: Resume from Most Recent Directory
```bash
pdm run batch-translate data/processed/raw_dataset --output-dir data/translations --resume
```
This automatically finds and resumes from the most recent translation directory.

##### Option 2: Use Custom Directory Name (Recommended)
```bash
# Start with custom name
pdm run batch-translate data/processed/raw_dataset --output-dir data/translations --output-name my_batch

# Resume from same directory
pdm run batch-translate data/processed/raw_dataset --output-dir data/translations --output-name my_batch
```

##### How Resume Works:
1. **Automatic Detection**: The system automatically detects existing translations in the target directory
2. **Skip Existing**: Already completed translations are skipped
3. **Continue Processing**: Translation resumes from the next incomplete item
4. **Preserve Progress**: All previously completed work is preserved

Example output when resuming:
```
Resuming from existing directory: data/translations/20241201_120000
Skipping existing translation: /path/to/report1.txt with llama3.1:8b (with_keywords)
Skipping existing translation: /path/to/report2.txt with llama3.1:8b (with_keywords)
Translating reports: 100%|██████████| 50/50 [10:30<00:00, 12.6s/it]

Translation Summary:
Total reports processed: 50
New translations completed: 25
Skipped existing translations: 25
Failed translations: 0
Total translations in file: 50
```

#### Python API Usage

For programmatic access to batch translation:

```python
from app.batch_translate import batch_translate, TranslationMode

# Basic batch translation
batch_translate(
    input_dataset_path="data/processed/raw_dataset",
    output_dataset_path="data/translations/batch_001",
    source_lang="de",
    target_lang="en",
    num_reports=100,
    mode=TranslationMode.WITH_KEYWORDS,
    models=["llama3.1:8b"],
    force_retranslate=False
)

# Resume interrupted translation
batch_translate(
    input_dataset_path="data/processed/raw_dataset",
    output_dataset_path="data/translations/batch_001",  # Same output path
    source_lang="de",
    target_lang="en",
    mode=TranslationMode.BOTH,
    models=["llama3.1:8b", "llama3.1:70b"]
)
```

### Available Scripts

The project includes several PDM scripts for common tasks:

```bash
# Dataset and translation
pdm run convert-dataset          # Convert raw text files to dataset
pdm run batch-translate          # Flexible batch translation with custom parameters
pdm run run-translation          # Predefined batch translation (150 reports, both modes, with dictionary fallback)
pdm run run-translation-no-dict  # Predefined batch translation without dictionary fallback

# Evaluation and analysis
pdm run evaluate-translations    # Evaluate translation quality using COMET metrics
pdm run create-excel-comparison  # Create Excel comparison files
pdm run analyze-words            # Analyze word frequency and failed translations
pdm run create-dictionary        # Create concept dictionary from failed translations
pdm run dictionary               # Manage concept dictionary (stats, search, test)

# Services and display
pdm run start-azure              # Start Azure Text Analytics container
pdm run display-results          # Display results in Streamlit interface
```

See all available scripts in `pyproject.toml`:

## Requirements

- Python 3.12
- Docker (for running Azure Text Analytics container)
- Azure Text Analytics API key and endpoint
- Langfuse account and API keys
- PDM package manager
- COMET evaluation library (installed automatically with `pdm install`)

## Open issues / TODO

- Set minimum confidence level for concept extraction to avoid false positives 
- Improve dataset result structure: One object per source text and one nested object per model (and another nested object per method (with/out keywords))
- Add list of manually collected improvements to prompt (added via string matching?)