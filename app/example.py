"""Example usage of the translation tool."""

from pprint import pprint

from config import settings
from translation import TranslationRequest, translate_text
from exceptions import TranslationError, ModelNotFoundError, ServiceConnectionError

def main():
    """Run translation example."""
    # Example medical text in German to translate to English
    text = """
    Der Patient zeigt Symptome einer akuten Bronchitis mit starkem Husten und Fieber. Ausserdem klagt er über Kopfschmerzen.
    Die Auskultation ergab Rasselgeräusche in beiden Lungenflügeln. 
    Wir haben eine Behandlung mit Antibiotika (Amoxicillin) und Hustenlöser begonnen.
    """
    
    # Create a translation request
    request = TranslationRequest(
        text=text,
        source_lang="de",  # ISO language code for German
        target_lang="en",  # ISO language code for English
        model=settings.DEFAULT_MODEL
    )
    
    try:
        # Perform the translation
        results = translate_text(request)
        
        print("Original text:")
        print(text)
        print("\nTranslated text:")
        print(results["translated_text"])
        
        print(f"\n=== Translation Statistics ===")
        print(f"Extracted concepts: {results['num_extracted_concepts']}")
        print(f"SNOMED translations: {results['num_snomed_translations']}")
        print(f"UMLS lookups required: {results['num_umls_lookups']}")
        
        if results['concepts_requiring_umls']:
            print(f"\n=== Concepts that Required UMLS Lookup ===")
            for concept in results['concepts_requiring_umls']:
                print(f"- Term: {concept['term']}")
                print(f"  Category: {concept['category']}")
                print(f"  UMLS CUI: {concept['umls_cui']}")
                print(f"  SNOMED Code: {concept['snomed_code']}")
                print(f"  Success: {concept['success']}")
                print()
        
        print(f"\n=== Full Results ===")
        pprint(results)
        
    except ModelNotFoundError as e:
        print(f"\nModel error: {str(e)}")
        return 1
    except ServiceConnectionError as e:
        print(f"\nService error: {str(e)}")
        return 1
    except TranslationError as e:
        print(f"\nTranslation error: {str(e)}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 