"""Service for interacting with Ollama LLM."""

from typing import Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from exceptions import ModelNotFoundError, ServiceConnectionError

class LLMService:
    """Service for interacting with Ollama."""
    
    def __init__(self):
        """Initialize the LLM service using configuration."""
        self.client = OpenAI(
            base_url=settings.OLLAMA_BASE_URL,
            #api_key='sk-c5dd5a8603f3488f839a341f2d526ce3'  # TODO: Move to env
           # api_key='AIzaSyBq2j5Bs2e-bryJr9TAsZnMWDJjzfDytOs'  # Google Cloud private account
            api_key='AIzaSyDIIsrjpvk3goSxN2I6OMkizHQRwvGpcRc' # Google Glouc bfh account
        )
    
    @retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.RETRY_BASE_WAIT, min=4, max=settings.RETRY_MAX_WAIT),
        reraise=True
    )
    def generate_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = settings.DEFAULT_TEMPERATURE
    ) -> Dict[str, Any]:
        """
        Generate a completion from the LLM.
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            model: Model to use
            temperature: Temperature for generation
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            ModelNotFoundError: If the specified model is not available
            ServiceConnectionError: If there's an error connecting to Ollama
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            return {
                "text": response.choices[0].message.content.strip(),
                "model": response.model,
                "usage": response.usage.model_dump(),
                "finish_reason": response.choices[0].finish_reason,
                "raw_response": response.model_dump()  # Full raw response from the model
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            if "model not found" in error_msg:
                raise ModelNotFoundError(f"Model '{model}' not found. Please pull it first using: ollama pull {model}")
            else:
                raise ServiceConnectionError(f"Error connecting to Ollama service: {str(e)}")

    def validate_model(self, model: str) -> bool:
        """
        Validate if a model exists and is available.
        
        Args:
            model: Name of the model to validate
            
        Returns:
            True if model is available, False otherwise
            
        Raises:
            ServiceConnectionError: If there's an error connecting to Ollama
        """
        try:
            self.generate_completion(
                system_prompt="Test prompt",
                user_prompt="Test",
                model=model
            )
            return True
        except ModelNotFoundError:
            return False

# Create a global instance
llm_service = LLMService() 