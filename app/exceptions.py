"""Custom exceptions for the translation system."""

class TranslationError(Exception):
    """Base exception for translation errors."""
    pass

class ModelNotFoundError(TranslationError):
    """Raised when an LLM model is not available."""
    pass

class ServiceConnectionError(TranslationError):
    """Raised when a service connection fails."""
    pass

class ExtractorError(TranslationError):
    """Raised when medical term extraction fails."""
    pass

class SnomedError(TranslationError):
    """Raised when SNOMED CT service encounters an error."""
    pass

class UMLSError(TranslationError):
    """Raised when UMLS service encounters an error."""
    pass 

class EvaluationError(Exception):
    """Raised when evaluation fails."""
    pass