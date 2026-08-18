"""
Translation module that uses LLMs and external services for text translation with keyword enhancement.
"""

import os
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from dotenv import load_dotenv
import prompts
from extractors import AzureHealthExtractor, ExtractedKeyword
from snomed_service import SnomedTranslationService
from umls_service import UMLSService
from llm_service import llm_service
from dictionary_service import get_concept_dictionary
from enhanced_dictionary_service import get_enhanced_dictionary
from exceptions import ModelNotFoundError, ServiceConnectionError, TranslationError

# Load environment variables
load_dotenv()

# Initialize services
snomed_service = SnomedTranslationService()
umls_service = UMLSService()

class TranslationRequest(BaseModel):
    """Model for translation requests."""
    text: str
    source_lang: str
    target_lang: str
    model: str = "gpt-oss:120b"  # Default model
    use_dictionary_fallback: bool = True  # Enable dictionary fallback by default
    use_dictionary_only: bool = False  # Use only verified dictionary translations, skip SNOMED/UMLS
    use_file_prompts: bool = False  # Use system prompts from files instead of simple hardcoded prompts

class TranslationResult(BaseModel):
    """Model for translation results including all intermediary data."""
    # Input data
    original_text: str
    source_lang: str
    target_lang: str
    
    # Extraction results
    extracted_concepts: List[Dict[str, Any]]
    num_extracted_concepts: int
    
    # UMLS lookup results - detailed tracking
    umls_lookups: List[Dict[str, Any]]
    concepts_requiring_umls: List[Dict[str, Any]]  # New field for tracking which concepts needed UMLS
    num_umls_lookups: int  # Count of concepts that required UMLS lookup
    
    # SNOMED CT results
    snomed_translations: List[Dict[str, Any]]
    num_snomed_translations: int
    
    # Translation results
    translated_text: str
    translation_metadata: Dict[str, Any]
    
    # Performance metrics
    timing: Dict[str, float]

def get_snomed_code(keyword: ExtractedKeyword) -> Dict[str, Any]:
    """
    Get SNOMED CT code either directly or via UMLS.
    
    Args:
        keyword: The keyword with its medical codes
        
    Returns:
        Dictionary containing lookup results and metadata
    """
    result = {
        "term": keyword.word,
        "normalized_text": keyword.normalized_text or keyword.word,
        "category": keyword.category,
        "confidence_score": keyword.confidence_score,
        "original_codes": keyword.codes,
        "snomed_code": None,
        "source": None,
        "success": False,
        "required_umls_lookup": False,
        "umls_cui": None,
        "lookup_timestamp": None
    }
    
    try:
        # First try to get SNOMED CT code directly
        if 'SNOMEDCT' in keyword.codes:
            result.update({
                "snomed_code": keyword.codes['SNOMEDCT'],
                "source": "direct",
                "success": True,
                "required_umls_lookup": False
            })
            return result
            
        # If not available but UMLS code exists, try to get SNOMED CT code via UMLS
        if 'UMLS' in keyword.codes:
            from datetime import datetime
            lookup_start = datetime.now()
            
            snomed_code = umls_service.get_snomed_code_from_umls(keyword.codes['UMLS'])
            
            result.update({
                "snomed_code": snomed_code,
                "source": "umls",
                "success": snomed_code is not None,
                "required_umls_lookup": True,
                "umls_cui": keyword.codes['UMLS'],
                "lookup_timestamp": lookup_start.isoformat()
            })
            return result
            
        # No SNOMED or UMLS codes available
        result.update({
            "source": "none",
            "required_umls_lookup": False
        })
        return result
        
    except Exception as e:
        result.update({
            "error": str(e),
            "required_umls_lookup": False
        })
        return result

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_keyword(keyword: ExtractedKeyword, source_lang: str, target_lang: str, use_dictionary_fallback: bool = True, use_dictionary_only: bool = False) -> Dict[str, Any]:
    """
    Translate a specific keyword using SNOMED CT service if available, with optional dictionary fallback.
    
    Args:
        keyword: The keyword with its medical codes to translate
        source_lang: Source language code
        target_lang: Target language code
        use_dictionary_fallback: Whether to use dictionary fallback if SNOMED fails
        use_dictionary_only: Whether to use only verified dictionary translations, skip SNOMED/UMLS
        
    Returns:
        Dictionary containing translation results and metadata
    """
    result = {
        "term": keyword.word,
        "normalized_text": keyword.normalized_text or keyword.word,
        "category": keyword.category,
        "confidence_score": keyword.confidence_score,
        "codes": keyword.codes,
        "translation": None,
        "success": False,
        "translation_source": None
    }
    
    try:
        # If dictionary-only mode is enabled, skip SNOMED/UMLS lookups entirely
        if use_dictionary_only:
            concept_dictionary = get_concept_dictionary()
            if concept_dictionary.is_available():
                dictionary_result = concept_dictionary.get_verified_translation_with_metadata(
                    keyword.word, target_lang
                )
                
                if dictionary_result and dictionary_result["verified"]:
                    result.update({
                        "translation": dictionary_result["translation"],
                        "success": True,
                        "translation_source": "dictionary_only",
                        "dictionary_metadata": {
                            "verified": dictionary_result["verified"],
                            "categories": dictionary_result["categories"],
                            "failure_count": dictionary_result["failure_count"],
                            "avg_confidence": dictionary_result["avg_confidence"]
                        },
                        "snomed_lookup": {
                            "term": keyword.word,
                            "normalized_text": keyword.normalized_text or keyword.word,
                            "category": keyword.category,
                            "confidence_score": keyword.confidence_score,
                            "original_codes": keyword.codes,
                            "snomed_code": None,
                            "source": "skipped_dictionary_only",
                            "success": False,
                            "required_umls_lookup": False,
                            "umls_cui": None,
                            "lookup_timestamp": None
                        }
                    })
                    return result
                else:
                    # No verified dictionary translation found
                    result.update({
                        "translation_source": "dictionary_only_no_verified",
                        "snomed_lookup": {
                            "term": keyword.word,
                            "normalized_text": keyword.normalized_text or keyword.word,
                            "category": keyword.category,
                            "confidence_score": keyword.confidence_score,
                            "original_codes": keyword.codes,
                            "snomed_code": None,
                            "source": "skipped_dictionary_only",
                            "success": False,
                            "required_umls_lookup": False,
                            "umls_cui": None,
                            "lookup_timestamp": None
                        }
                    })
                    return result
            else:
                # Dictionary not available in dictionary-only mode
                result.update({
                    "translation_source": "dictionary_only_unavailable",
                    "snomed_lookup": {
                        "term": keyword.word,
                        "normalized_text": keyword.normalized_text or keyword.word,
                        "category": keyword.category,
                        "confidence_score": keyword.confidence_score,
                        "original_codes": keyword.codes,
                        "snomed_code": None,
                        "source": "skipped_dictionary_only",
                        "success": False,
                        "required_umls_lookup": False,
                        "umls_cui": None,
                        "lookup_timestamp": None
                    }
                })
                return result
        
        # Normal mode: First try SNOMED CT translation
        snomed_lookup = get_snomed_code(keyword)
        result["snomed_lookup"] = snomed_lookup
        
        if snomed_lookup["snomed_code"]:
            translation = snomed_service.get_preferred_term(
                snomed_lookup["snomed_code"],
                target_lang
            )
            if translation:
                result.update({
                    "translation": translation,
                    "success": True,
                    "translation_source": "snomed"
                })
                return result
        
        # If SNOMED translation failed, try dictionary fallback (if enabled)
        if use_dictionary_fallback:
            concept_dictionary = get_concept_dictionary()
            if concept_dictionary.is_available():
                dictionary_result = concept_dictionary.get_translation_with_metadata(
                    keyword.word, target_lang
                )
                
                if dictionary_result:
                    result.update({
                        "translation": dictionary_result["translation"],
                        "success": True,
                        "translation_source": "dictionary",
                        "dictionary_metadata": {
                            "verified": dictionary_result["verified"],
                            "categories": dictionary_result["categories"],
                            "failure_count": dictionary_result["failure_count"],
                            "avg_confidence": dictionary_result["avg_confidence"]
                        }
                    })
                    return result
        
        # If both SNOMED and dictionary failed, return failure
        result["translation_source"] = "none"
        return result
        
    except Exception as e:
        result["error"] = str(e)
        result["translation_source"] = "error"
        return result

def translate_text_dictionary_only(request: TranslationRequest) -> Dict[str, Any]:
    """
    Translate text using only direct string matching from the enhanced dictionary.
    This bypasses concept extraction, UMLS retrieval, and SNOMED calls.
    
    Args:
        request: TranslationRequest containing the text and language information
        
    Returns:
        Dictionary containing translation results and metadata
    """
    try:
        # Initialize result structure
        result = {
            "original_text": request.text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "model_used": request.model,
            "extracted_concepts": [],
            "umls_lookups": [],
            "concepts_requiring_umls": [],
            "num_umls_lookups": 0,
            "snomed_translations": [],
            "timing": {},
            "translation_metadata": {},
            "prompt_details": {
                "keyword_context": "",
                "final_prompt": "",
                "system_prompt": ""
            },
            "dictionary_matches": []
        }
        
        # 1. Find dictionary matches using direct string matching
        enhanced_dictionary = get_enhanced_dictionary()
        if not enhanced_dictionary.is_available():
            raise TranslationError("Enhanced dictionary is not available for dictionary-only translation")
        
        dictionary_matches = enhanced_dictionary.find_matches_in_text(request.text, request.target_lang)
        result["dictionary_matches"] = dictionary_matches
        
        # 2. Create keyword context from dictionary matches
        translations = {}
        for match in dictionary_matches:
            translations[match['term']] = match['translation']
        
        # Create keyword context (deduplicated)
        unique_translations = {}
        for match in dictionary_matches:
            term = match['term']
            translation = match['translation']
            # Use lowercase key for case-insensitive deduplication
            term_key = term.lower()
            if term_key not in unique_translations:
                unique_translations[term_key] = {
                    'original_term': term,
                    'translation': translation
                }
        
        keyword_context = "\n".join([
            f"- {data['original_term']} ({prompts.get_language_name(request.source_lang)}) → {data['translation']} ({prompts.get_language_name(request.target_lang)})"
            for data in unique_translations.values()
        ])
        
        # Store keyword context
        result["prompt_details"]["keyword_context"] = keyword_context
        
        # 3. Create the translation prompt with keyword context
        user_prompt = prompts.get_prompt(
            prompts.FINAL_TRANSLATION_PROMPT,
            source_lang=prompts.get_language_name(request.source_lang),
            target_lang=prompts.get_language_name(request.target_lang),
            keyword_context=keyword_context,
            text=request.text
        )
        
        # Load system prompt based on flag
        if request.use_file_prompts:
            system_prompt_content = prompts.load_system_prompt_from_file("with_keywords")
            system_prompt = {
                "name": "system_prompt_with_keywords",
                "content": system_prompt_content
            }
        else:
            system_prompt = prompts.get_prompt(prompts.FINAL_TRANSLATOR_SYSTEM)
        
        # Store prompts
        result["prompt_details"]["final_prompt"] = user_prompt["content"]
        result["prompt_details"]["system_prompt"] = system_prompt["content"]
        
        # 4. Get final translation from LLM
        llm_response = llm_service.generate_completion(
            system_prompt=system_prompt["content"],
            user_prompt=user_prompt["content"],
            model=request.model
        )
        
        # Store translation and metadata
        result["translated_text"] = llm_response["text"]
        result["translation_metadata"].update({
            "model": llm_response["model"],
            "usage": llm_response["usage"],
            "finish_reason": llm_response["finish_reason"],
            "raw_response": llm_response["raw_response"]
        })
        
        # Update statistics
        result["num_extracted_concepts"] = len(dictionary_matches)
        result["num_snomed_translations"] = len([m for m in dictionary_matches if m.get('verified', False)])
        
        return result
        
    except Exception as e:
        raise TranslationError(f"Error in dictionary-only translation: {str(e)}")

def translate_text(request: TranslationRequest) -> Dict[str, Any]:
    """
    Translate text using LLM with keyword enhancement.
    
    Args:
        request: TranslationRequest containing the text and language information
        
    Returns:
        Dictionary containing all translation results, intermediary data, and metadata
    """
    # If dictionary-only mode is enabled, use direct string matching
    if request.use_dictionary_only:
        return translate_text_dictionary_only(request)
    
    try:
        # Initialize result structure
        result = {
            "original_text": request.text,
            "source_lang": request.source_lang,
            "target_lang": request.target_lang,
            "model_used": request.model,
            "extracted_concepts": [],
            "umls_lookups": [],
            "concepts_requiring_umls": [],
            "num_umls_lookups": 0,
            "snomed_translations": [],
            "timing": {},
            "translation_metadata": {},
            "prompt_details": {
                "keyword_context": "",
                "final_prompt": "",
                "system_prompt": ""
            }
        }
        
        # 1. Extract important keywords using Azure Text Analytics for Health
        extractor = AzureHealthExtractor(request.source_lang)
        keywords = extractor.extract_keywords(request.text)
        
        # Store extraction results
        result["extracted_concepts"] = [
            {
                "term": k.word,
                "normalized_text": k.normalized_text or k.word,
                "category": k.category,
                "confidence_score": k.confidence_score,
                "codes": k.codes
            } for k in keywords
        ]
        result["num_extracted_concepts"] = len(keywords)
        
        # 2. Translate each keyword
        translations = {}
        for keyword in keywords:
            try:
                translation_result = translate_keyword(
                    keyword, 
                    request.source_lang, 
                    request.target_lang,
                    request.use_dictionary_fallback,
                    request.use_dictionary_only
                )
                
                # Track UMLS lookups and concepts that required UMLS
                snomed_lookup = translation_result["snomed_lookup"]
                if snomed_lookup["required_umls_lookup"]:
                    result["umls_lookups"].append(snomed_lookup)
                    result["concepts_requiring_umls"].append({
                        "term": keyword.word,
                        "normalized_text": keyword.normalized_text or keyword.word,
                        "category": keyword.category,
                        "confidence_score": keyword.confidence_score,
                        "original_codes": keyword.codes,
                        "umls_cui": snomed_lookup.get("umls_cui"),
                        "snomed_code": snomed_lookup.get("snomed_code"),
                        "lookup_timestamp": snomed_lookup.get("lookup_timestamp"),
                        "success": snomed_lookup.get("success", False)
                    })
                
                # Store SNOMED translation results
                result["snomed_translations"].append(translation_result)
                
                # Store translation for prompt
                translations[keyword.word] = translation_result["translation"] if translation_result["translation"] else keyword.word
                
            except RetryError:
                print(f"Failed to translate keyword after retries: {keyword.word}")
                translations[keyword.word] = keyword.word
        
        # Update statistics
        result["num_snomed_translations"] = len([t for t in result["snomed_translations"] if t["success"]])
        result["num_umls_lookups"] = len(result["concepts_requiring_umls"])
        
        # 3. Create the translation prompt with keyword context (deduplicated)
        # Create a set of unique terms to avoid duplicates in the prompt (case-insensitive)
        unique_translations = {}
        for k in keywords:
            if translations.get(k.word) != k.word:  # Only include terms that were actually translated
                # Use lowercase key for case-insensitive deduplication
                term_key = k.word.lower()
                if term_key not in unique_translations:
                    unique_translations[term_key] = {
                        'original_term': k.word,
                        'translation': translations[k.word]
                    }
        
        keyword_context = "\n".join([
            f"- {data['original_term']} ({prompts.get_language_name(request.source_lang)}) → {data['translation']} ({prompts.get_language_name(request.target_lang)})"
            for data in unique_translations.values()
        ])
        
        # Store keyword context
        result["prompt_details"]["keyword_context"] = keyword_context
        
        user_prompt = prompts.get_prompt(
            prompts.FINAL_TRANSLATION_PROMPT,
            source_lang=prompts.get_language_name(request.source_lang),
            target_lang=prompts.get_language_name(request.target_lang),
            keyword_context=keyword_context,
            text=request.text
        )
        
        # Load system prompt based on flag
        if request.use_file_prompts:
            system_prompt_content = prompts.load_system_prompt_from_file("with_keywords")
            system_prompt = {
                "name": "system_prompt_with_keywords",
                "content": system_prompt_content
            }
        else:
            system_prompt = prompts.get_prompt(prompts.FINAL_TRANSLATOR_SYSTEM)
        
        # Store prompts
        result["prompt_details"]["final_prompt"] = user_prompt["content"]
        result["prompt_details"]["system_prompt"] = system_prompt["content"]
        
        # 4. Get final translation from LLM
        llm_response = llm_service.generate_completion(
            system_prompt=system_prompt["content"],
            user_prompt=user_prompt["content"],
            model=request.model
        )
        
        # Store translation and metadata
        result["translated_text"] = llm_response["text"]
        result["translation_metadata"].update({
            "model": llm_response["model"],
            "usage": llm_response["usage"],
            "finish_reason": llm_response["finish_reason"],
            "raw_response": llm_response["raw_response"]  # Store full raw response
        })
        
        return result
        
    except Exception as e:
        raise Exception(f"Error translating text: {str(e)}")

def analyze_umls_lookup_patterns(translation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze UMLS lookup patterns across multiple translation results.
    
    Args:
        translation_results: List of translation results from translate_text()
        
    Returns:
        Dictionary containing analysis of UMLS lookup patterns
    """
    analysis = {
        "total_translations": len(translation_results),
        "total_concepts": 0,
        "total_umls_lookups": 0,
        "concepts_by_category": {},
        "umls_lookup_success_rate": 0.0,
        "concepts_requiring_umls": [],
        "direct_snomed_available": 0,
        "no_codes_available": 0
    }
    
    all_concepts_requiring_umls = []
    
    for result in translation_results:
        analysis["total_concepts"] += result.get("num_extracted_concepts", 0)
        analysis["total_umls_lookups"] += result.get("num_umls_lookups", 0)
        
        # Collect all concepts that required UMLS lookup
        concepts_requiring_umls = result.get("concepts_requiring_umls", [])
        all_concepts_requiring_umls.extend(concepts_requiring_umls)
        
        # Analyze by category
        for concept in concepts_requiring_umls:
            category = concept.get("category", "unknown")
            if category not in analysis["concepts_by_category"]:
                analysis["concepts_by_category"][category] = 0
            analysis["concepts_by_category"][category] += 1
    
    # Calculate success rate
    successful_umls_lookups = sum(1 for concept in all_concepts_requiring_umls if concept.get("success", False))
    if analysis["total_umls_lookups"] > 0:
        analysis["umls_lookup_success_rate"] = successful_umls_lookups / analysis["total_umls_lookups"]
    
    # Count direct SNOMED availability and no codes
    for result in translation_results:
        for translation in result.get("snomed_translations", []):
            snomed_lookup = translation.get("snomed_lookup", {})
            source = snomed_lookup.get("source", "none")
            if source == "direct":
                analysis["direct_snomed_available"] += 1
            elif source == "none":
                analysis["no_codes_available"] += 1
    
    analysis["concepts_requiring_umls"] = all_concepts_requiring_umls
    
    return analysis

def translate_without_keywords(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str,
    use_file_prompts: bool = False
) -> Dict[str, Any]:
    """
    Translate text directly without keyword extraction.
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        model: Model to use for translation
        use_file_prompts: Whether to use system prompts from files instead of simple hardcoded prompts
        
    Returns:
        Translation results
        
    Raises:
        ModelNotFoundError: If the specified model is not available
        ServiceConnectionError: If there's an error connecting to services
        TranslationError: For other translation-related errors
    """
    # Load system prompt based on flag
    if use_file_prompts:
        system_prompt_content = prompts.load_system_prompt_from_file("without_keywords")
        system_prompt = {
            "name": "system_prompt_without_keywords",
            "content": system_prompt_content
        }
    else:
        system_prompt = prompts.get_prompt(prompts.FINAL_TRANSLATOR_SYSTEM)
    
    user_prompt = prompts.get_prompt(
        prompts.DIRECT_TRANSLATION_PROMPT,
        source_lang=prompts.get_language_name(source_lang),
        target_lang=prompts.get_language_name(target_lang),
        text=text
    )
    
    try:
        # Get translation directly from LLM
        response = llm_service.generate_completion(
            system_prompt=system_prompt["content"],
            user_prompt=user_prompt["content"],
            model=model
        )
        
        # Create result structure matching the with-keywords format
        result = {
            "original_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "translated_text": response["text"],
            "translation_mode": "without_keywords",
            "model_used": model,
            "translation_metadata": {
                "model": response["model"],
                "usage": response["usage"],
                "finish_reason": response["finish_reason"],
                "raw_response": response["raw_response"]  # Store full raw response
            },
            "prompt_details": {
                "system_prompt": system_prompt["content"],
                "user_prompt": user_prompt["content"]
            }
        }
        
        return result
        
    except (ModelNotFoundError, ServiceConnectionError) as e:
        raise  # Re-raise these exceptions as they should be handled at a higher level
    except Exception as e:
        raise TranslationError(f"Error in direct translation: {str(e)}")