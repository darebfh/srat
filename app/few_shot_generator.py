"""
Few-shot example generation module for medical translation system prompts.

This module processes sample medical reports and generates few-shot examples
that can be manually added to system prompts to improve translation quality.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

try:
    from .translation import translate_text, TranslationRequest
    from .batch_translate import translate_without_keywords
    from .prompts import get_language_name
except ImportError:
    # For direct execution
    from translation import translate_text, TranslationRequest
    from batch_translate import translate_without_keywords
    from prompts import get_language_name


@dataclass
class FewShotExample:
    """Represents a few-shot example with source text, translation, and metadata."""
    source_text: str
    target_text: str
    source_lang: str
    target_lang: str
    translation_mode: str  # "with_keywords" or "without_keywords"
    keyword_context: str = ""  # For with_keywords mode


class FewShotGenerator:
    """Generates few-shot examples from sample medical reports using actual translation modules."""
    
    def __init__(self, sample_reports_dir: str = "assets/sample_reports"):
        """
        Initialize the few-shot generator.
        
        Args:
            sample_reports_dir: Directory containing sample medical reports
        """
        self.sample_reports_dir = Path(sample_reports_dir)
        self.reports = []
        self.examples = []
        
    def load_sample_reports(self) -> List[Dict[str, str]]:
        """
        Load all sample reports from the directory.
        
        Returns:
            List of dictionaries containing report content and metadata
        """
        if not self.sample_reports_dir.exists():
            raise FileNotFoundError(f"Sample reports directory not found: {self.sample_reports_dir}")
        
        reports = []
        for report_file in self.sample_reports_dir.glob("*.txt"):
            with open(report_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            reports.append({
                "filename": report_file.name,
                "content": content
            })
        
        self.reports = reports
        return reports
    
    def generate_with_keywords_example(
        self, 
        source_text: str, 
        source_lang: str = "en", 
        target_lang: str = "de",
        model: str = "gpt-oss:120b"
    ) -> Optional[FewShotExample]:
        """
        Generate a few-shot example using the with_keywords translation mode.
        
        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            model: LLM model to use for translation
            
        Returns:
            FewShotExample object or None if translation fails
        """
        try:
            # Create translation request
            request = TranslationRequest(
                text=source_text,
                source_lang=source_lang,
                target_lang=target_lang,
                model=model,
                use_dictionary_fallback=True
            )
            
            # Perform translation with keywords
            result = translate_text(request)
            
            if not result.get("translated_text"):
                print(f"Warning: With-keywords translation failed for text: {source_text[:100]}...")
                return None
            
            # Create few-shot example
            example = FewShotExample(
                source_text=source_text,
                target_text=result["translated_text"],
                source_lang=source_lang,
                target_lang=target_lang,
                translation_mode="with_keywords",
                keyword_context=result.get("prompt_details", {}).get("keyword_context", "")
            )
            
            return example
            
        except Exception as e:
            print(f"Error generating with-keywords example: {e}")
            return None
    
    def generate_without_keywords_example(
        self, 
        source_text: str, 
        source_lang: str = "en", 
        target_lang: str = "de",
        model: str = "gpt-oss:120b"
    ) -> Optional[FewShotExample]:
        """
        Generate a few-shot example using the without_keywords translation mode.
        
        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            model: LLM model to use for translation
            
        Returns:
            FewShotExample object or None if translation fails
        """
        try:
            # Perform translation without keywords
            result = translate_without_keywords(
                text=source_text,
                source_lang=source_lang,
                target_lang=target_lang,
                model=model
            )
            
            if not result.get("translated_text"):
                print(f"Warning: Without-keywords translation failed for text: {source_text[:100]}...")
                return None
            
            # Create few-shot example
            example = FewShotExample(
                source_text=source_text,
                target_text=result["translated_text"],
                source_lang=source_lang,
                target_lang=target_lang,
                translation_mode="without_keywords",
                keyword_context=""
            )
            
            return example
            
        except Exception as e:
            print(f"Error generating without-keywords example: {e}")
            return None
    
    def generate_both_variants(
        self, 
        source_text: str, 
        source_lang: str = "en", 
        target_lang: str = "de",
        model: str = "gpt-oss:120b"
    ) -> Tuple[Optional[FewShotExample], Optional[FewShotExample]]:
        """
        Generate both with_keywords and without_keywords variants for a text.
        
        Args:
            source_text: Source text to translate
            source_lang: Source language code
            target_lang: Target language code
            model: LLM model to use for translation
            
        Returns:
            Tuple of (with_keywords_example, without_keywords_example)
        """
        with_keywords_example = self.generate_with_keywords_example(
            source_text, source_lang, target_lang, model
        )
        
        without_keywords_example = self.generate_without_keywords_example(
            source_text, source_lang, target_lang, model
        )
        
        return with_keywords_example, without_keywords_example
    
    def create_sample_examples(self) -> List[FewShotExample]:
        """
        Create sample few-shot examples using the loaded reports and actual translation modules.
        
        Returns:
            List of FewShotExample objects
        """
        if not self.reports:
            self.load_sample_reports()
        
        examples = []
        
        for report in self.reports:
            print(f"Generating examples for {report['filename']}...")
            
            # Generate both variants
            with_keywords_example, without_keywords_example = self.generate_both_variants(
                source_text=report["content"],
                source_lang="en",
                target_lang="de",
                model="gpt-oss:120b"
            )
            
            if with_keywords_example:
                examples.append(with_keywords_example)
                print(f"✓ Generated with_keywords example for {report['filename']}")
            else:
                print(f"✗ Failed to generate with_keywords example for {report['filename']}")
            
            if without_keywords_example:
                examples.append(without_keywords_example)
                print(f"✓ Generated without_keywords example for {report['filename']}")
            else:
                print(f"✗ Failed to generate without_keywords example for {report['filename']}")
        
        self.examples = examples
        return examples
    
    
    def generate_system_prompt_with_keywords(self, target_lang: str = "de") -> str:
        """
        Generate a system prompt with keywords and few-shot examples.
        
        Args:
            target_lang: Target language for examples
            
        Returns:
            Complete system prompt string
        """
        if not self.examples:
            self.create_sample_examples()
        
        # Get with_keywords examples only
        with_keywords_examples = [ex for ex in self.examples if ex.translation_mode == "with_keywords"]
        
        if not with_keywords_examples:
            return "No with_keywords examples available."
        
        # Format examples
        examples_section = ""
        for i, example in enumerate(with_keywords_examples, 1):
            keyword_context = f"\nMedical term translations:\n{example.keyword_context}" if example.keyword_context else ""
            examples_section += f"""Example {i}:
Source ({example.source_lang}): {example.source_text}
{keyword_context}
Target ({example.target_lang}): {example.target_text}

---"""
        
        system_prompt = f"""You are a professional medical translator. Provide only the translation without any explanations.

# Few-Shot Examples for Medical Translation with Keywords

The following examples demonstrate proper medical translation from English to {get_language_name(target_lang)} with medical term enhancement:

{examples_section}

## Translation Guidelines

Based on these examples, follow these principles:
1. Maintain clinical accuracy and use standard medical terminology
2. Preserve the formal, professional tone of medical reports
3. Keep anatomical and diagnostic terms precise and consistent
4. Maintain the structured format of medical reports
5. Use the provided medical term translations when available
6. Ensure all medical codes and technical terms are properly translated

"""
        
        return system_prompt
    
    def generate_system_prompt_without_keywords(self, target_lang: str = "de") -> str:
        """
        Generate a system prompt without keywords, only with few-shot examples.
        
        Args:
            target_lang: Target language for examples
            
        Returns:
            Complete system prompt string
        """
        if not self.examples:
            self.create_sample_examples()
        
        # Get without_keywords examples only
        without_keywords_examples = [ex for ex in self.examples if ex.translation_mode == "without_keywords"]
        
        if not without_keywords_examples:
            return "No without_keywords examples available."
        
        # Format examples
        examples_section = ""
        for i, example in enumerate(without_keywords_examples, 1):
            examples_section += f"""Example {i}:
Source ({example.source_lang}): {example.source_text}
Target ({example.target_lang}): {example.target_text}

---"""
        
        system_prompt = f"""You are a professional medical translator. Provide only the translation without any explanations.

# Few-Shot Examples for Medical Translation

The following examples demonstrate proper medical translation from English to {get_language_name(target_lang)}:

{examples_section}

## Translation Guidelines

Based on these examples, follow these principles:
1. Maintain clinical accuracy and use standard medical terminology
2. Preserve the formal, professional tone of medical reports
3. Keep anatomical and diagnostic terms precise and consistent
4. Maintain the structured format of medical reports
5. Ensure all medical codes and technical terms are properly translated

"""
        
        return system_prompt
    
    def save_system_prompts(self, target_lang: str = "de") -> None:
        """
        Save both system prompt variants to files.
        
        Args:
            target_lang: Target language for examples
        """
        if not self.examples:
            self.create_sample_examples()
        
        # Generate and save with_keywords system prompt
        with_keywords_prompt = self.generate_system_prompt_with_keywords(target_lang)
        with open("system_prompt_with_keywords.txt", 'w', encoding='utf-8') as f:
            f.write(with_keywords_prompt)
        print("System prompt with keywords saved to: system_prompt_with_keywords.txt")
        
        # Generate and save without_keywords system prompt
        without_keywords_prompt = self.generate_system_prompt_without_keywords(target_lang)
        with open("system_prompt_without_keywords.txt", 'w', encoding='utf-8') as f:
            f.write(without_keywords_prompt)
        print("System prompt without keywords saved to: system_prompt_without_keywords.txt")
    


def main():
    """Main function to generate system prompts."""
    generator = FewShotGenerator()
    
    # Load sample reports
    reports = generator.load_sample_reports()
    print(f"Loaded {len(reports)} sample reports")
    
    # Generate examples using actual translation modules
    print("Generating few-shot examples...")
    examples = generator.create_sample_examples()
    
    if examples:
        print(f"Generated {len(examples)} examples")
        
        # Save system prompts
        generator.save_system_prompts(target_lang="de")
        print("Done!")
    else:
        print("No examples were generated.")


if __name__ == "__main__":
    main()