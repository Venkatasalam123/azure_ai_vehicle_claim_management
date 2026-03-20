import os
from dotenv import load_dotenv
from pathlib import Path

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential


def analyze_claim_document(
    file_path: str,
    model_id: str = "prebuilt-read"
) -> dict:
    """
    Analyze an insurance claim document from local file system.
    Supports PDFs and images.
    Returns structured text and table data.
    """

    # Load environment variables from parent directory
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)


    # -----------------------------
    # Read config (NON-SECRET)
    # -----------------------------
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url:
        raise ValueError(
            "DOCUMENT_INTELLIGENCE_ENDPOINT or KEYVAULT_URL not set"
        )

    # -----------------------------
    # Authenticate to Key Vault
    # -----------------------------
    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential
    )

    # -----------------------------
    # Fetch Document Intelligence Key
    # -----------------------------
    docintelligence_key = secret_client.get_secret(
        "docintelligence-key"
    ).value

    # -----------------------------
    # Create Document Intelligence Client
    # -----------------------------
    doc_client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(docintelligence_key)
    )

    # -----------------------------
    # Read local document
    # -----------------------------
    with open(file_path, "rb") as f:
        document_bytes = f.read()

    # -----------------------------
    # Analyze document
    # -----------------------------
    poller = doc_client.begin_analyze_document(
        model_id=model_id,
        analyze_request=AnalyzeDocumentRequest(
            bytes_source=document_bytes
        )
    )

    result = poller.result()

    # -----------------------------
    # Extract required details
    # -----------------------------
    extracted_data = {
        "model_used": model_id,

        "pages": [
            {
                "page_number": page.page_number,
                "lines": [
                    line.content for line in (page.lines or [])
                ]
            }
            for page in (result.pages or [])
        ],

        "tables": [
            {
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": [
                    {
                        "row": cell.row_index,
                        "column": cell.column_index,
                        "content": cell.content
                    }
                    for cell in table.cells
                ]
            }
            for table in (result.tables or [])
        ]
    }

    return extracted_data
