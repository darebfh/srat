# Concept Dictionary System

The concept dictionary system provides a fallback mechanism for medical term translations when SNOMED CT translation fails. It automatically creates a dictionary from failed translations and integrates seamlessly with the translation pipeline.

## Overview

The dictionary system consists of three main components:

1. **Dictionary Converter** - Converts failed translations to an editable CSV dictionary
2. **Dictionary Service** - Loads and queries the concept dictionary
3. **Translation Integration** - Automatically uses dictionary as fallback when SNOMED fails

## Quick Start

### 1. Generate Dictionary from Failed Translations

```bash
# Analyze failed translations and create dictionary
pdm run analyze-words data/translations/20241201_120000/translations.json --analyze-failed

# Convert failed translations to editable CSV dictionary
pdm run create-dictionary translations_failed_translations_summary.json -o assets/dictionaries/concept_dictionary.csv
```

### 2. Edit the Dictionary

Open `assets/dictionaries/concept_dictionary.csv` in your preferred editor and:
- Replace placeholder translations with actual translations
- Mark verified entries as `TRUE` in the `verified` column
- Add notes for context

### 3. Use Dictionary in Translation Pipeline

The dictionary is automatically used as a fallback when SNOMED CT translation fails. No additional configuration needed!

## Dictionary Structure

The CSV dictionary contains the following columns:

| Column | Description |
|--------|-------------|
| `term` | Original medical term |
| `normalized_text` | Normalized version of the term |
| `categories` | Medical categories (comma-separated) |
| `failure_count` | Number of times this term failed translation |
| `avg_confidence` | Average confidence score from extraction |
| `translation_de` | German translation |
| `translation_fr` | French translation |
| `translation_es` | Spanish translation |
| `translation_it` | Italian translation |
| `notes` | Additional notes |
| `verified` | Whether the translation has been manually verified |

## Dictionary Management

### View Dictionary Statistics

```bash
pdm run dictionary stats
```

Output:
```
📊 Dictionary Statistics
==============================
Total entries: 1145
Verified entries: 0
Dictionary path: /path/to/assets/dictionaries/concept_dictionary.csv

Translations by language:
  DE: 80
  FR: 80
  ES: 80
  IT: 80
```

### Search Terms in Dictionary

```bash
pdm run dictionary search "chest" --limit 5
```

Output:
```
🔍 Search results for 'chest' (showing 5 results)
============================================================
 1. ❌ thorax
    Normalized: Chest
    Categories: BodyStructure
    Translations:
      DE: Brustkorb
      FR: thorax
      ES: tórax
      IT: torace
```

### Test Translations

```bash
pdm run dictionary test "chest" de
```

Output:
```
✅ Translation found for 'chest' → 'DE'
Translation: Brustkorb
Verified: ❌
Categories: BodyStructure
Failure count: 1
Avg confidence: 0.96
```

## Translation Pipeline Integration

The dictionary is automatically integrated into the translation pipeline and can be controlled via a command-line flag:

### Translation Flow

1. **SNOMED CT Translation** - First attempt using SNOMED CT service
2. **Dictionary Fallback** - If SNOMED fails and dictionary is enabled, try dictionary lookup
3. **Return Result** - Return translation with source metadata

### Dictionary Fallback Control

The dictionary fallback can be enabled or disabled using the `--no-dictionary-fallback` flag:

```bash
# Enable dictionary fallback (default)
pdm run run-translation

# Disable dictionary fallback
pdm run run-translation-no-dict

# Or use the flag directly
python app/batch_translate.py data/processed/raw_dataset --no-dictionary-fallback
```

### Translation Sources

The system tracks translation sources:

- `snomed` - Translation from SNOMED CT service
- `dictionary` - Translation from concept dictionary
- `none` - No translation found
- `error` - Translation error occurred

### Enhanced Translation Results

When using the dictionary, translation results include additional metadata:

```json
{
  "translation": "Brustkorb",
  "success": true,
  "translation_source": "dictionary",
  "dictionary_metadata": {
    "verified": false,
    "categories": "BodyStructure",
    "failure_count": 1,
    "avg_confidence": 0.96
  }
}
```

## Programmatic Usage

### Using the Dictionary Service

```python
from app.dictionary_service import get_concept_dictionary, translate_with_dictionary

# Get dictionary instance
dictionary = get_concept_dictionary()

# Check if dictionary is available
if dictionary.is_available():
    # Get translation
    translation = dictionary.get_translation("chest", "de")
    print(f"Translation: {translation}")
    
    # Get translation with metadata
    result = dictionary.get_translation_with_metadata("chest", "de")
    if result:
        print(f"Translation: {result['translation']}")
        print(f"Verified: {result['verified']}")
        print(f"Categories: {result['categories']}")

# Convenience function
translation = translate_with_dictionary("chest", "de")
print(f"Translation: {translation}")
```

### Dictionary Statistics

```python
from app.dictionary_service import get_concept_dictionary

dictionary = get_concept_dictionary()
stats = dictionary.get_stats()

print(f"Total entries: {stats['total_entries']}")
print(f"Verified entries: {stats['verified_entries']}")
print(f"Translations by language: {stats['translations_by_language']}")
```

### Search Functionality

```python
from app.dictionary_service import get_concept_dictionary

dictionary = get_concept_dictionary()
matches = dictionary.search_terms("chest", limit=5)

for match in matches:
    print(f"Term: {match['term']}")
    print(f"Verified: {match['verified']}")
    print(f"Translations: {match['translations']}")
```

## Dictionary Workflow

### 1. Initial Setup

```bash
# Run translation analysis to identify failed translations
pdm run analyze-words data/translations/20241201_120000/translations.json --analyze-failed

# Convert to editable dictionary
pdm run create-dictionary translations_failed_translations_summary.json
```

### 2. Dictionary Editing

1. Open `assets/dictionaries/concept_dictionary.csv` in a spreadsheet editor
2. Review the most frequently failed terms (high `failure_count`)
3. Replace placeholder translations with actual translations
4. Mark verified entries as `TRUE`
5. Add context notes if needed

### 3. Dictionary Validation

```bash
# Check dictionary statistics
pdm run dictionary stats

# Test specific translations
pdm run dictionary test "chest" de
pdm run dictionary test "lung" fr

# Search for terms
pdm run dictionary search "pain" --limit 10
```

### 4. Integration Testing

The dictionary is automatically used in the translation pipeline. To verify integration:

1. Run new translations
2. Check that previously failing terms now succeed
3. Monitor translation source metadata

## Best Practices

### Dictionary Maintenance

1. **Regular Updates**: Update dictionary after each translation batch
2. **Quality Control**: Verify translations before marking as verified
3. **Category Focus**: Prioritize high-frequency failure terms
4. **Language Consistency**: Ensure consistent terminology across languages

### Translation Quality

1. **Medical Accuracy**: Use authoritative medical dictionaries
2. **Context Awareness**: Consider medical context when translating
3. **Verification**: Have medical professionals review translations
4. **Documentation**: Add notes for complex or ambiguous terms

### Performance Optimization

1. **Verified First**: Prioritize verified translations
2. **Frequency-Based**: Focus on high-frequency terms
3. **Category Grouping**: Group related terms for batch processing
4. **Regular Cleanup**: Remove outdated or incorrect entries

## File Locations

- **Dictionary CSV**: `assets/dictionaries/concept_dictionary.csv`
- **Failed Translations**: `translations_failed_translations_summary.json`
- **Dictionary Service**: `app/dictionary_service.py`
- **Dictionary Converter**: `app/dictionary_converter.py`
- **Dictionary CLI**: `app/dictionary_cli.py`

## CLI Commands Summary

| Command | Description |
|---------|-------------|
| `pdm run create-dictionary <input.json>` | Convert failed translations to CSV dictionary |
| `pdm run dictionary stats` | Show dictionary statistics |
| `pdm run dictionary search <query>` | Search terms in dictionary |
| `pdm run dictionary test <term> <lang>` | Test translation of a term |
| `pdm run analyze-words <file.json> --analyze-failed` | Analyze failed translations |
| `pdm run run-translation` | Run translation with dictionary fallback (default) |
| `pdm run run-translation-no-dict` | Run translation without dictionary fallback |

## Troubleshooting

### Dictionary Not Loading

```bash
# Check if dictionary file exists
ls -la assets/dictionaries/concept_dictionary.csv

# Check dictionary statistics
pdm run dictionary stats
```

### Translation Not Found

```bash
# Search for the term
pdm run dictionary search "your_term"

# Check if it's in the failed translations
grep -i "your_term" translations_failed_translations_summary.json
```

### Performance Issues

- Dictionary is loaded once and cached
- Large dictionaries (>10k entries) may take a few seconds to load
- Consider splitting very large dictionaries by category

## Integration with Existing Workflow

The dictionary system integrates seamlessly with your existing translation workflow:

1. **No Code Changes**: Dictionary fallback is automatic
2. **Backward Compatible**: Existing translations continue to work
3. **Enhanced Metadata**: Additional translation source information
4. **Gradual Improvement**: Dictionary improves over time as you add translations

## Future Enhancements

Potential improvements for the dictionary system:

1. **Auto-translation**: Use LLMs to suggest translations
2. **Category-specific dictionaries**: Separate dictionaries by medical specialty
3. **Version control**: Track dictionary changes over time
4. **Collaborative editing**: Multiple users editing the same dictionary
5. **Quality metrics**: Track translation accuracy and usage
