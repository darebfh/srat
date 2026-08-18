"""
Service for translating medical terms using local SNOMED CT service.
"""

import requests
from typing import Optional, Dict
from pydantic import BaseModel

from config import settings
from exceptions import SnomedError, ServiceConnectionError

class SnomedDescription(BaseModel):
    """Model for SNOMED CT description."""
    active: bool
    term: str
    lang: str
    acceptabilityMap: Dict[str, str]
    type: str

class SnomedTranslationService:
    """Service for translating using local SNOMED CT service."""
    
    def __init__(self):
        """Initialize the SNOMED service using configuration."""
        self.base_url = settings.SNOMED_BASE_URL.rstrip('/')

    def get_preferred_term(self, concept_id: str, target_lang: str) -> Optional[str]:
        """
        Get the preferred term for a concept in the target language.
        
        Args:
            concept_id: SNOMED CT concept ID
            target_lang: Target language code (e.g., 'de' for German)
            
        Returns:
            The preferred term in the target language if found, None otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to SNOMED service
            SnomedError: If there's an error processing the SNOMED response
        """
        try:
            response = requests.get(
                f"{self.base_url}/MAIN/concepts/{concept_id}/descriptions",
                headers={
                    'accept': 'application/json',
                    'Accept-Language': 'en-X-900000000000509007,en-X-900000000000508004,en'
                }
            )
            response.raise_for_status()
            
            # Get the response data
            data = response.json()
            descriptions_data = data.get('conceptDescriptions', [])
            
            # Parse each description in the response
            descriptions = []
            for desc in descriptions_data:
                try:
                    descriptions.append(SnomedDescription(
                        active=desc.get('active', False),
                        term=desc.get('term', ''),
                        lang=desc.get('lang', ''),
                        acceptabilityMap=desc.get('acceptabilityMap', {}),
                        type=desc.get('type', '')
                    ))
                except Exception as e:
                    raise SnomedError(f"Error parsing SNOMED description: {e}")
            
            # Filter for active descriptions in target language with PREFERRED acceptability
            preferred_desc = next(
                (desc for desc in descriptions 
                 if desc.active and 
                 desc.lang == target_lang and 
                 any(value == "PREFERRED" for value in desc.acceptabilityMap.values())
                ),
                None
            )
            
            return preferred_desc.term if preferred_desc else None
            
        except requests.RequestException as e:
            raise ServiceConnectionError(f"Error connecting to SNOMED service: {e}")
        except SnomedError:
            raise
        except Exception as e:
            raise SnomedError(f"Error processing SNOMED response: {e}")

    def validate_service(self) -> bool:
        """
        Validate if the SNOMED service is available and accessible.
        
        Returns:
            True if service is available, False otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to SNOMED service
        """
        try:
            # Try to access a simple endpoint to validate the service
            response = requests.get(
                f"{self.base_url}/MAIN/concepts/",
                headers={'accept': 'application/json'},
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            raise ServiceConnectionError(f"Error connecting to SNOMED service: {e}")
        except Exception as e:
            raise ServiceConnectionError(f"Error validating SNOMED service: {e}")