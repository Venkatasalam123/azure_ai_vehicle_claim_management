import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.language.conversations import ConversationAnalysisClient
import os
from pathlib import Path
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential




def analyze_with_clu(user_text: str):

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # Environment variables: LANGUAGE_KEY, LANGUAGE_ENDPOINT, LANGUAGE_PROJECT_NAME, LANGUAGE_DEPLOYMENT_NAME
    CLU_ENDPOINT = os.getenv("CLU_ENDPOINT")
    CLU_PROJECT_NAME = os.getenv("CLU_PROJECT_NAME")
    CLU_DEPLOYMENT_NAME = os.getenv("CLU_DEPLOYMENT_NAME")

    keyvault_url = os.getenv("KEYVAULT_URL")

    if not CLU_ENDPOINT or not keyvault_url:
        raise ValueError("CLU_ENDPOINT or KEYVAULT_URL not set")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential,
    )

    CLU_KEY = secret_client.get_secret("clu-key").value

    clu_client = ConversationAnalysisClient(
        endpoint=CLU_ENDPOINT,
        credential=AzureKeyCredential(CLU_KEY)
    )

    request = {
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "id": "1",
                "participantId": "user",
                "text": user_text
            }
        },
        "parameters": {
            "projectName": CLU_PROJECT_NAME,
            "deploymentName": CLU_DEPLOYMENT_NAME,
            "stringIndexType": "TextElement_V8"
        }
    }

    response = clu_client.analyze_conversation(request)
    result = response["result"]["prediction"]

    top_intent = result["topIntent"]

    # intents is a LIST
    intents = result["intents"]
    confidence = next(
        intent["confidenceScore"]
        for intent in intents
        if intent["category"] == top_intent
    )

    # entities is also a LIST
    entities = {}
    for entity in result.get("entities", []):
        entities[entity["category"]] = entity["text"]

    return {
        "intent": top_intent,
        "confidence": confidence,
        "entities": entities
    }

# result = analyze_with_clu("Show me the damaged vehicles in the claim")
# print(result)