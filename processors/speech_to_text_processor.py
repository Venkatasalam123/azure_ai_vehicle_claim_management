from azure.cognitiveservices import speech as speechsdk
import os
from pathlib import Path
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from translator_processor import *
from conversation_language_understanding import *
from content_safety import *
from router import *

def speech_to_text_once():
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    endpoint = os.getenv("SPEECH_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url:
        raise ValueError("SPEECH_ENDPOINT or KEYVAULT_URL not set")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential,
    )

    key = secret_client.get_secret("speech-key").value


    speech_config = speechsdk.SpeechConfig(endpoint=endpoint, subscription=key)

    # Auto language detection (add/remove languages as needed)
    auto_lang = speechsdk.AutoDetectSourceLanguageConfig(
        languages=["en-US", "ta-IN", "hi-IN"]
    )

    # Mic input
    audio_config = speechsdk.AudioConfig(use_default_microphone=True)

    # Recognizer
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        auto_detect_source_language_config=auto_lang,
        audio_config=audio_config
    )

    print("🎤 Speak now...")
    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_lang = result.properties[
            speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult
        ]
        return {
            "text": result.text,
            "language": detected_lang
        }

    if result.reason == speechsdk.ResultReason.NoMatch:
        raise RuntimeError("No speech recognized.")

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Canceled: {details.reason} | {details.error_details}")

# out = speech_to_text_once()
# print(out)

# translation_result = translate_to_english(out["text"])
# print(translation_result)

# content_safety_result = moderate_input(translation_result['translated_text'])
# print(content_safety_result)

# if content_safety_result["allowed"] == False:
#     print("❌ Content blocked")
# else:
#     clu_input = (
#         translation_result["translated_text"]
#         if translation_result["detected_language"] != "en"
#         else out["text"]
#     )

#     clu_result = analyze_with_clu(clu_input)
#     print("CLU Result:", clu_result)

#     router_result = route_request(
#         clu_result=clu_result,
#         original_query=clu_input
#     )

#     print("Router Result:", router_result)