"""
Service for interacting with UMLS Terminology Services (UTS) REST API.
"""

import re
from typing import Optional
import requests

from config import settings
from exceptions import UMLSError, ServiceConnectionError

class UMLSService:
    """Service for UMLS API interactions."""
    
    def __init__(self):
        """Initialize the UMLS service using configuration."""
        self.api_key = settings.UMLS_API_KEY
        if not self.api_key:
            raise ValueError("UMLS API key is required")
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        
    def get_snomed_code_from_umls(self, umls_cui: str) -> Optional[str]:
        """
        Get SNOMED CT code from UMLS CUI.
        
        Args:
            umls_cui: UMLS Concept Unique Identifier (CUI)
            
        Returns:
            SNOMED CT code if found, None otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to UMLS
            UMLSError: If there's an error processing the UMLS response
        """
        try:
            response = requests.get(
                f"{self.base_url}/content/current/CUI/{umls_cui}/atoms",
                params={
                    "sabs": "SNOMEDCT_US",
                    "apiKey": self.api_key
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Find the entry with termType "PT" (Preferred Term)
            fn_entry = next(
                (item for item in data.get("result", [])
                 if item.get("termType") == "PT"),
                None
            )
            
            if not fn_entry:
                return None
                
            # Extract SNOMED CT ID from the code URL
            code_url = fn_entry.get("code", "")
            match = re.search(r"/SNOMEDCT_US/(\d+)$", code_url)
            return match.group(1) if match else None
            
        except requests.RequestException as e:
            raise ServiceConnectionError(f"Error connecting to UMLS service: {e}")
        except Exception as e:
            raise UMLSError(f"Error processing UMLS response: {e}")

    def validate_service(self) -> bool:
        """
        Validate if the UMLS service is available and accessible.
        
        Returns:
            True if service is available, False otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to UMLS
        """
        try:
            # Try to access a simple endpoint to validate the service
            response = requests.get(
                f"{self.base_url}/search/current",
                params={
                    "string": "test",
                    "apiKey": self.api_key
                },
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            raise ServiceConnectionError(f"Error connecting to UMLS service: {e}")
        except Exception as e:
            raise ServiceConnectionError(f"Error validating UMLS service: {e}") 