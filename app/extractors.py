"""
Keyword extraction module using Azure Text Analytics for Health.
"""

from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from pydantic import BaseModel

from config import settings
from exceptions import ExtractorError, ServiceConnectionError

# Target code system for medical concepts
TARGET_CODE_SYSTEMS = ["SNOMEDCT", "UMLS"]

class ExtractedKeyword(BaseModel):
    """Model for extracted keywords with their SNOMED CT code."""
    word: str
    normalized_text: Optional[str] = None
    codes: Dict[str, str]
    confidence_score: float
    category: str

class AzureHealthExtractor:
    """Extracts medical keywords using Azure Text Analytics for Health."""
    
    def __init__(self, language: str):
        """
        Initialize the extractor.
        
        Args:
            language: Language code (en, es, fr, de, it, pt)
            
        Raises:
            ServiceConnectionError: If there's an error connecting to Azure
            ValueError: If required configuration is missing
        """
        self.client = self._authenticate_client(language)
        
    def _authenticate_client(self, language: str) -> TextAnalyticsClient:
        """
        Authenticate with Azure Text Analytics.
        
        Args:
            language: Default language for the client
            
        Returns:
            Authenticated TextAnalyticsClient
            
        Raises:
            ServiceConnectionError: If there's an error connecting to Azure
            ValueError: If required configuration is missing
        """
        key = settings.LANGUAGE_BILLING_KEY
        endpoint = settings.LANGUAGE_ENDPOINT
        
        if not key or not endpoint:
            raise ValueError("LANGUAGE_BILLING_KEY and LANGUAGE_ENDPOINT must be set")
        
        try:
            credential = AzureKeyCredential(key)
            return TextAnalyticsClient(
                endpoint=endpoint,
                credential=credential,
                default_language=language,
                default_country_hint="CH"
            )
        except Exception as e:
            raise ServiceConnectionError(f"Error connecting to Azure service: {e}")
    
    @staticmethod
    def _get_codes(entity) -> Dict[str, str]:
        """
        Extract SNOMED CT code from entity data sources.
        
        Args:
            entity: Azure Text Analytics entity
            
        Returns:
            Dictionary of codes by system
        """
        codes = {}
        if entity.data_sources is not None:
            for entry in entity.data_sources:
                if entry.name in TARGET_CODE_SYSTEMS:
                    codes[entry.name] = entry.entity_id
        return codes
    
    def extract_keywords(self, text: str, confidence_threshold: float = None) -> List[ExtractedKeyword]:
        """
        Extract medical keywords that have SNOMED CT codes.
        
        Args:
            text: The text to analyze
            confidence_threshold: Minimum confidence score for entities (defaults to config value)
            
        Returns:
            List of extracted keywords with their SNOMED CT codes
            
        Raises:
            ExtractorError: If there's an error in keyword extraction
            ServiceConnectionError: If there's an error connecting to Azure
        """
        try:
            # Use provided threshold or default from config
            if confidence_threshold is None:
                confidence_threshold = settings.AZURE_CONFIDENCE_THRESHOLD
            
            # Get entities from Azure Text Analytics for Health
            poller = self.client.begin_analyze_healthcare_entities([text])
            response = poller.result()
            docs = [doc for doc in response if not doc.is_error]
                
            result = docs[0]  # We only sent one document
            
            # Check for errors
            if result.is_error:
                raise ExtractorError(f"Error in healthcare entity recognition: {result.error}")
                
            # Filter and convert entities to keywords
            keywords = []
            for entity in result.entities:
                codes = self._get_codes(entity)
                # Only include entities that have a SNOMED CT code AND meet confidence threshold
                if codes and entity.confidence_score >= confidence_threshold:
                    keyword = ExtractedKeyword(
                        word=entity.text,
                        normalized_text=entity.normalized_text,
                        codes=codes,
                        confidence_score=entity.confidence_score,
                        category=entity.category
                    )
                    keywords.append(keyword)
            
            return keywords
            
        except Exception as e:
            if isinstance(e, ExtractorError):
                raise
            raise ExtractorError(f"Error extracting keywords: {e}")

    def validate_service(self) -> bool:
        """
        Validate if the Azure Text Analytics service is available and accessible.
        
        Returns:
            True if service is available, False otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to Azure
        """
        try:
            # Try to analyze a simple test text to validate the service
            test_text = "Patient has diabetes."
            poller = self.client.begin_analyze_healthcare_entities([test_text])
            response = poller.result()
            docs = [doc for doc in response if not doc.is_error]
            
            if not docs:
                raise ServiceConnectionError("Azure Text Analytics service returned no valid results")
            
            return True
        except Exception as e:
            raise ServiceConnectionError(f"Error validating Azure Text Analytics service: {e}") 