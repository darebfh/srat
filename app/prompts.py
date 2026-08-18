"""
Prompt management module.
"""

import os
from typing import Dict, Any

# Language code to full name mapping
LANGUAGE_MAPPING = {
    "en": "English",
    "de": "German"
}

def get_language_name(lang_code: str) -> str:
    """
    Convert language code to full language name.
    
    Args:
        lang_code: ISO language code (e.g., "en", "de")
        
    Returns:
        Full language name (e.g., "English", "German")
    """
    return LANGUAGE_MAPPING.get(lang_code.lower(), lang_code.upper())

# Define prompt templates
FINAL_TRANSLATOR_SYSTEM = {
    "name": "final_translator_system",
    "prompt": "You are a professional medical translator. Provide only the translation without any explanations."
}

FINAL_TRANSLATION_PROMPT = {
    "name": "final_translation",
    "prompt": """Translate the following medical text from {source_lang} to {target_lang}.

Important medical terms and their translations:
{keyword_context}

Text to translate:
{text}

Translated text:
"""
}

DIRECT_TRANSLATION_PROMPT = {
    "name": "direct_translation",
    "prompt": """Translate the following medical text from {source_lang} to {target_lang}.

Text to translate:
{text}

Translated text:
"""
}

def load_system_prompt_from_file(mode: str) -> str:
    """
    Load system prompt from file based on translation mode.
    
    Args:
        mode: Translation mode - "with_keywords" or "without_keywords"
        
    Returns:
        System prompt content as string
        
    Raises:
        FileNotFoundError: If the system prompt file doesn't exist
        ValueError: If mode is not supported
    """
    if mode not in ["with_keywords", "without_keywords"]:
        raise ValueError(f"Unsupported mode: {mode}. Must be 'with_keywords' or 'without_keywords'")
    
    # Get the directory where this module is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the assets/system_prompts directory
    prompts_dir = os.path.join(current_dir, "..", "assets", "system_prompts")
    
    if mode == "with_keywords":
        filename = "system_prompt_with_keywords.txt"
    else:  # without_keywords
        filename = "system_prompt_without_keywords.txt"
    
    file_path = os.path.join(prompts_dir, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"System prompt file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_prompt(prompt_template: Dict[str, str], **kwargs) -> Dict[str, str]:
    """
    Get a prompt with variables replaced.
    
    Args:
        prompt_template: The prompt template dictionary
        **kwargs: Variables to replace in the prompt
        
    Returns:
        Dictionary containing the prompt name and rendered content
    """
    return {
        "name": prompt_template["name"],
        "content": prompt_template["prompt"].format(**kwargs) if kwargs else prompt_template["prompt"]
    }