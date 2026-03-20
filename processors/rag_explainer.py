from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient


def get_openai_client():
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential,
    )

    OPENAI_KEY = secret_client.get_secret("openai-key").value

    return AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview"
    )


def generate_summary_from_docs(documents: list):
    """
    documents = [
        {
            "id": "...",
            "content": "...",
            "source_type": "document|image",
            "created_at": "..."
        }
    ]
    """

    if not documents:
        return {
            "summary": "No relevant claim documents were found for the given criteria.",
            "confidence": 0.3
        }

    # Truncate to avoid token explosion
    context_blocks = []
    for doc in documents[:5]:
        text = doc.get("content", "")
        if text:
            context_blocks.append(text[:800])

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are an insurance claims analyst.

Based ONLY on the following claim information, summarize the common repair issues.
Do not add information that is not present.

CLAIM DATA:
{context}
"""

    client = get_openai_client()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You summarize insurance claim data."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return {
        "summary": response.choices[0].message.content.strip(),
        "confidence": 0.8
    }
