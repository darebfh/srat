# Word Frequency Analysis Module

This module analyzes word frequency in completed translation JSON files, filtering out short words like articles and prepositions to focus on meaningful content.

## Features

- **Smart Filtering**: Automatically excludes short words (configurable minimum length)
- **Stop Word Removal**: Filters out common articles, prepositions, and conjunctions in multiple languages (English, German, French, Spanish, Italian)
- **Multiple Output Formats**: Save results as JSON, CSV, or plain text
- **Command Line Interface**: Easy-to-use CLI for quick analysis
- **Programmatic API**: Use the `WordAnalyzer` class in your own code

## Quick Start

### Command Line Usage

```bash
# Basic analysis (saves to same directory as input file)
pdm run analyze-words data/translations/20241201_120000/translations.json
# Output: data/translations/20241201_120000/translations_word_frequency.json

# Custom settings with automatic output path
python app/word_analysis.py data/translations/20241201_120000/translations.json \
    --top-n 50 \
    --min-length 4 \
    --format csv \
    --print-results
# Output: data/translations/20241201_120000/translations_word_frequency.csv

# Custom output path
python app/word_analysis.py data/translations/20241201_120000/translations.json \
    --output custom_word_frequency.csv \
    --format csv

# Include stop words in analysis
python app/word_analysis.py data/translations/20241201_120000/translations.json \
    --no-stop-words \
    --top-n 200

# Analyze failed translations (concepts that couldn't be translated)
python app/word_analysis.py data/translations/20241201_120000/translations.json \
    --analyze-failed \
    --failed-format csv
# Creates: data/translations/20241201_120000/translations_failed_translations.csv
# And: data/translations/20241201_120000/translations_failed_translations_summary.csv
```

### Command Line Options

- `input_file`: Path to the translation JSON file (required)
- `-o, --output`: Output file path (default: same directory as input file with _word_frequency suffix)
- `-f, --format`: Output format - json, csv, or txt (default: json)
- `-n, --top-n`: Number of top words to return (default: 100)
- `--min-length`: Minimum word length to include (default: 3)
- `--no-stop-words`: Include common stop words in analysis
- `--print-results`: Print results to console
- `--analyze-failed`: Also analyze failed translations and save to separate file
- `--failed-output`: Output file path for failed translations analysis (default: same directory as input file)
- `--failed-format`: Output format for failed translations analysis - json, csv, or txt (default: json)

### Programmatic Usage

```python
from app.word_analysis import WordAnalyzer

# Create analyzer with default settings
analyzer = WordAnalyzer()

# Or with custom settings
analyzer = WordAnalyzer(
    min_word_length=4,
    exclude_common_words=False
)

# Analyze word frequency
results = analyzer.get_most_common_words(
    "data/translations/20241201_120000/translations.json",
    top_n=50
)

# Save results (using default path in same directory as input)
output_path = analyzer.get_default_output_path("data/translations/20241201_120000/translations.json", "json")
analyzer.save_results(results, output_path, "json")

# Or save to custom path
analyzer.save_results(results, "custom_output.json", "json")

# Or work with the data directly
for word, frequency in results:
    print(f"{word}: {frequency}")

# Analyze failed translations
failed_analysis = analyzer.analyze_failed_translations("data/translations/20241201_120000/translations.json")
print(f"Success rate: {failed_analysis['success_rate']}%")

# Get summary of most frequently failed terms
failed_summary = analyzer.get_failed_translations_summary("data/translations/20241201_120000/translations.json")
for term, count, categories, normalized_text, avg_confidence in failed_summary[:5]:
    print(f"{term}: failed {count} times")
```

## Input Format

The module expects JSON files containing an array of translation objects. Each translation object should have an `original_text` field:

```json
[
  {
    "original_text": "The patient presents with chest pain.",
    "translated_text": "Der Patient präsentiert sich mit Brustschmerzen.",
    "source_lang": "en",
    "target_lang": "de",
    "model_used": "gpt-4",
    "translation_mode": "with_keywords"
  },
  {
    "original_text": "Medical examination reveals elevated blood pressure.",
    "translated_text": "Die medizinische Untersuchung zeigt erhöhten Blutdruck.",
    "source_lang": "en",
    "target_lang": "de",
    "model_used": "gpt-4",
    "translation_mode": "with_keywords"
  }
]
```

## Output Formats

### JSON Format
```json
[
  {"word": "patient", "frequency": 45},
  {"word": "medical", "frequency": 32},
  {"word": "examination", "frequency": 28}
]
```

### CSV Format
```csv
word,frequency
patient,45
medical,32
examination,28
```

### Text Format
```
patient: 45
medical: 32
examination: 28
```

## Filtering Logic

The module applies several filters to focus on meaningful words:

1. **Length Filter**: Words shorter than the minimum length (default: 3 characters) are excluded
2. **Stop Word Filter**: Common function words are excluded:
   - Articles: the, a, an, der, die, das, le, la, el, la, etc.
   - Prepositions: in, on, at, zu, für, dans, sur, en, con, etc.
   - Conjunctions: and, or, but, und, oder, et, ou, y, o, etc.
   - Pronouns: I, you, he, ich, du, er, je, tu, il, yo, tú, él, etc.
3. **Numeric Filter**: Pure numbers are excluded
4. **Punctuation Removal**: All punctuation is removed before analysis

## Supported Languages

Stop word filtering supports:
- English
- German
- French
- Spanish
- Italian

## Examples

### Example 1: Basic Analysis
```bash
pdm run analyze-words data/translations/20241201_120000/translations.json
```

### Example 2: Medical Text Analysis
```bash
python app/word_analysis.py data/translations/medical_reports.json \
    --min-length 4 \
    --top-n 200 \
    --format csv \
    --output medical_word_frequency.csv
```

### Example 3: Include All Words
```bash
python app/word_analysis.py data/translations/reports.json \
    --no-stop-words \
    --min-length 2 \
    --top-n 500
```

## Failed Translations Analysis

The module can also analyze concepts that failed to be translated using the SNOMED CT service. This helps identify:

- **Medical terms without translations**: Concepts that couldn't be translated
- **Translation success rates**: Overall performance of the translation system
- **Problematic categories**: Which types of medical concepts fail most often
- **Frequently failing terms**: Terms that consistently fail translation

### Failed Translations Output

When using `--analyze-failed`, the module creates two files:

1. **Detailed Analysis** (`*_failed_translations.json/csv/txt`):
   - Complete list of all failed translations
   - Includes context (file path, patient ID, etc.)
   - Error details and confidence scores

2. **Summary** (`*_failed_translations_summary.json/csv/txt`):
   - Terms grouped by failure frequency
   - Categories and confidence scores
   - Most problematic terms

### Example Failed Translations Analysis

```bash
# Analyze both word frequency and failed translations
python app/word_analysis.py data/translations/20241201_120000/translations.json \
    --analyze-failed \
    --failed-format csv \
    --print-results
```

This will show output like:
```
Failed Translations Summary:
Total concepts analyzed: 1250
Failed translations: 89
Success rate: 92.88%

Top 10 most frequently failed terms:
 1. rare_disease_term     (failed 5 times, avg confidence: 0.85)
 2. specialized_medication (failed 3 times, avg confidence: 0.72)
 3. anatomical_variant    (failed 2 times, avg confidence: 0.91)
```

## Integration with Translation Pipeline

This module is designed to work with the translation results from the batch translation system:

1. Run batch translation: `pdm run run-translation`
2. Analyze word frequency: `pdm run analyze-words data/translations/[timestamp]/translations.json`
3. Analyze failed translations: `pdm run analyze-words data/translations/[timestamp]/translations.json --analyze-failed`
4. Use results for further analysis or reporting

## Error Handling

The module includes comprehensive error handling:
- File not found errors
- Invalid JSON format errors
- Empty or malformed translation data
- Missing `original_text` fields

## Performance

- Efficiently processes large translation files
- Uses Python's `Counter` for fast frequency counting
- Memory-efficient text processing
- Supports files with thousands of translations
