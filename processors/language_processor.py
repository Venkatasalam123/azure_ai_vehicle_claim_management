import os
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


def _get_language_client() -> TextAnalyticsClient:
    """
    Create and return a TextAnalyticsClient using config from .env and Key Vault.
    """
    # Load environment variables from local .env (same pattern as other processors)
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("LANGUAGE_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url:
        raise ValueError("LANGUAGE_ENDPOINT or KEYVAULT_URL not set")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential,
    )

    key = secret_client.get_secret("language-key").value

    return TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )


def analyze_claim_language(text: str) -> dict:
    """
    Analyze claim text using Azure AI Language to extract key phrases and entities.

    Returns a dict:
        {
            "key_phrases": [str, ...],
            "entities": [str, ...]
        }
    """
    if not text or not text.strip():
        return {
            "key_phrases": [],
            "entities": [],
        }

    client = _get_language_client()
    documents = [text]

    entities_result = client.recognize_entities(documents)[0]
    key_phrases_result = client.extract_key_phrases(documents)[0]

    entities = [entity.text for entity in entities_result.entities]
    key_phrases = list(key_phrases_result.key_phrases)

    return {
        "key_phrases": key_phrases,
        "entities": entities,
    }


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using Azure AI Language service.
    
    Returns a dict:
        {
            "sentiment": "positive" | "negative" | "neutral" | "mixed",
            "confidence_scores": {
                "positive": float,
                "neutral": float,
                "negative": float
            },
            "overall_score": float  # The highest confidence score
        }
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "confidence_scores": {
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0
            },
            "overall_score": 1.0
        }
    
    client = _get_language_client()
    documents = [text]
    
    sentiment_result = client.analyze_sentiment(documents, show_opinion_mining=True)[0]
    
    sentiment = sentiment_result.sentiment.lower()
    confidence_scores = {
        "positive": sentiment_result.confidence_scores.positive,
        "neutral": sentiment_result.confidence_scores.neutral,
        "negative": sentiment_result.confidence_scores.negative
    }
    
    # Get the highest confidence score
    overall_score = max(confidence_scores.values())
    
    return {
        "sentiment": sentiment,
        "confidence_scores": confidence_scores,
        "overall_score": overall_score
    }