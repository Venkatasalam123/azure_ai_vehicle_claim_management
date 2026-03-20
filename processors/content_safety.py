from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
import os
from pathlib import Path
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential


env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

CONTENT_SAFETY_ENDPOINT = os.getenv("CONTENT_SAFETY_ENDPOINT")
keyvault_url = os.getenv("KEYVAULT_URL")

if not CONTENT_SAFETY_ENDPOINT or not keyvault_url:
    raise ValueError("CONTENT_SAFETY_ENDPOINT or KEYVAULT_URL not set")

credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

secret_client = SecretClient(
    vault_url=keyvault_url,
    credential=credential,
)

CONTENT_SAFETY_KEY = secret_client.get_secret("content-safety-key").value


cs_client = ContentSafetyClient(
    endpoint=CONTENT_SAFETY_ENDPOINT,
    credential=AzureKeyCredential(CONTENT_SAFETY_KEY)
)

def moderate_input(text: str):

    options = AnalyzeTextOptions(
        text=text,
        categories=["Hate", "SelfHarm", "Sexual", "Violence"]
    )

    result = cs_client.analyze_text(options=options)

    flags = []

    for a in result.categories_analysis:
        if (
            a.category == "Violence" and a.severity >= 3
        ) or (
            a.category != "Violence" and a.severity >= 3
        ):
            flags.append({
                "category": a.category,
                "severity": a.severity
            })

    return {
        "allowed": not bool(flags),
        "flags": flags
    }



def moderate_output(text: str):
    options = AnalyzeTextOptions(
        text=text,
        categories=["Hate", "SelfHarm", "Sexual", "Violence"]
    )

    result = cs_client.analyze_text(options=options)

    flags = []

    for a in result.categories_analysis:
        # Stricter rules for OUTPUT
        if a.severity >= 3:
            flags.append({
                "category": a.category,
                "severity": a.severity
            })

    return {
        "allowed": not bool(flags),
        "flags": flags
    }

# print(moderate_input("I want to hurt someone"))
# print(moderate_input("I will kill you"))
# print(moderate_input("I am going to stab someone with a knife"))

