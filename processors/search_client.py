import os
from dotenv import load_dotenv
from pathlib import Path
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Load .env from parent directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

def get_search_client():
    endpoint = os.getenv("AISEARCH_ENDPOINT")
    index_name = os.getenv("AISEARCH_INDEX_NAME")
    admin_key = os.getenv("AISEARCH_ADMIN_KEY")

    if not endpoint or not index_name or not admin_key:
        raise ValueError("Azure AI Search configuration missing")

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key)
    )
