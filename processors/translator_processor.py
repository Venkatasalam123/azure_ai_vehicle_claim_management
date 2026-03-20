import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from azure.ai.translation.text import TextTranslationClient, TranslatorCredential
from azure.ai.translation.text.models import InputTextItem
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient


# Load .env from project root for local development.
# On Render, secrets should be provided via environment variables.
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


@lru_cache(maxsize=1)
def get_translator_key() -> str:
    """
    Fetch the Translator API key from Azure Key Vault.
    """
    keyvault_url = os.getenv("KEYVAULT_URL")
    if not keyvault_url:
        raise ValueError("KEYVAULT_URL not set")

    secret_name = os.getenv("TRANSLATOR_KEY_SECRET_NAME", "translator-key")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(vault_url=keyvault_url, credential=credential)
    return secret_client.get_secret(secret_name).value


def translate_to_english(text: str):
    TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"
    TRANSLATOR_REGION = "eastus"

    translator_key = get_translator_key()
    credential = TranslatorCredential(
        key=translator_key,
        region=TRANSLATOR_REGION
    )

    client = TextTranslationClient(
        endpoint=TRANSLATOR_ENDPOINT,
        credential=credential
    )

    text_items = [InputTextItem(text=text)]

    response = client.translate(
        content=text_items,
        to=["en"]
    )

    result = response[0]
    detected_lang = result.detected_language.language
    translated_text = result.translations[0].text

    return {
        "original_text": text,
        "detected_language": detected_lang,
        "translated_text": translated_text
    }



# user_input = "சேதமடைந்த வாகனங்களை காட்டு"
# translation_result = translate_to_english(user_input)

# print(translation_result)

# if translation_result["detected_language"] != "en":
#     clu_input = translation_result["translated_text"]
# else:
#     clu_input = user_input

# print(clu_input)
# result = analyze_with_clu(clu_input)
# print(f"result = {result}")
