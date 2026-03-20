from .search_client import get_search_client
from .search_query import build_search_filter
from .custom_question_answering import custom_qa_handler
from .rag import retrieve_context, build_rag_prompt
from .content_safety import moderate_output
from openai import AzureOpenAI
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient

CONFIDENCE_THRESHOLD = 0.6

def route_request(clu_result: dict, original_query: str):
    """
    Deterministic router for Phase 3
    """

    intent = clu_result.get("intent")
    confidence = clu_result.get("confidence", 0)
    entities = clu_result.get("entities", {})

    # 1️⃣ Low-confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        response = keyword_fallback_handler(original_query)
    # 2️⃣ Intent-based routing
    elif intent in ["search_image", "search_document", "search_mixed"]:
        response = search_handler(intent, entities)
    elif intent == "faq":
        response = custom_qa_handler(original_query)
    elif intent == "explain":
        response = rag_handler(original_query)
    # 3️⃣ Default fallback
    else:
        response = unsupported_intent_handler(intent)

    # 🛡 OUTPUT CONTENT SAFETY
    if "answer" in response:
        safety = moderate_output(response["answer"])
        print("Content Safety Check:", safety)
        if not safety["allowed"]:
            return {
                "action": response.get("action"),
                "answer": "The response could not be shown due to content safety restrictions."
            }
    
    return response


def search_handler(intent, entities, top_k=5):
    print(f'intent = {intent}')
    print(f'entities = {entities}')
    print(f'top_k = {top_k}')

    search_client = get_search_client()

    filter_expression = build_search_filter(intent, entities)
    print(f'filter_expression = {filter_expression}')

    results = search_client.search(
        search_text="*",              # entity-driven search
        filter=filter_expression,
        top=top_k,
        query_type="semantic",
        semantic_configuration_name="claims-semantic-config"
    )

    output = []

    for doc in results:
        output.append({
            "id": doc.get("id"),
            "source_type": doc.get("source_type"),
            "source_file": doc.get("source_file"),
            "content": doc.get("content"),
            "image_url": doc.get("image_url"),
            "document_url": doc.get("document_url"),
            "created_at": doc.get("created_at")
        })

    return {
        "action": "search",
        "intent": intent,
        "filters": filter_expression,
        "results": output
    }


# def custom_qa_handler(query):
#     return {
#         "action": "custom_qa",
#         "query": query,
#         "message": "Custom QA handler invoked"
#     }


def rag_handler(question: str):

    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

    keyvault_url = os.getenv("KEYVAULT_URL")

    if not OPENAI_ENDPOINT or not keyvault_url:
        raise ValueError("OPENAI_ENDPOINT or KEYVAULT_URL not set")

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


    client = AzureOpenAI(
    api_key=OPENAI_KEY,
    azure_endpoint=OPENAI_ENDPOINT,
    api_version="2024-12-01-preview"
    )

    context = retrieve_context(question)

    if not context.strip():
        return {
            "action": "explain",
            "answer": "I do not have enough information to answer this question."
        }

    prompt = build_rag_prompt(context, question)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful insurance assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return {
        "action": "explain",
        "answer": response.choices[0].message.content
    }


def keyword_fallback_handler(query):
    return {
        "action": "fallback_search",
        "query": query,
        "message": "Fallback keyword search invoked"
    }


def unsupported_intent_handler(intent):
    return {
        "action": "unsupported",
        "intent": intent,
        "message": "Unsupported intent"
    }

