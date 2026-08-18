"""Centralized configuration management."""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable mapping."""
    
    # API Keys and Endpoints
    OPENAI_API_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGUAGE_BILLING_KEY: Optional[str] = None
    LANGUAGE_BILLING_ENDPOINT: Optional[str] = None
    LANGUAGE_ENDPOINT: Optional[str] = None
    UMLS_API_KEY: Optional[str] = None
    
    # LLM Configuration
    DEFAULT_MODEL: str = "gemma3:4b"
    DEFAULT_TEMPERATURE: float = 0.3
    
    # Service URLs
    #OLLAMA_BASE_URL: str = "https://inference.mlmp.ti.bfh.ch/api/v1"
    OLLAMA_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/" # Attention: Paid account required so that MIMIC data is not shared with Google
    SNOMED_BASE_URL: str = "http://localhost:8080"
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_BASE_WAIT: int = 1
    RETRY_MAX_WAIT: int = 10
    
    # Model Configuration
    LANGFUSE_HOST: str = "http://localhost:3000"
    
    # Azure Text Analytics Configuration
    AZURE_CONFIDENCE_THRESHOLD: float = 0.8
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow"
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

# Create a global settings instance
settings = get_settings() 