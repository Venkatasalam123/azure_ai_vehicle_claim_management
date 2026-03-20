from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential
import os
from pathlib import Path
from azure.keyvault.secrets import SecretClient
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential



def custom_qa_handler(question: str):
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    # Environment variables: LANGUAGE_KEY, LANGUAGE_ENDPOINT, LANGUAGE_PROJECT_NAME, LANGUAGE_DEPLOYMENT_NAME
    LANGUAGE_ENDPOINT = os.getenv("CLU_ENDPOINT")
    CQA_PROJECT_NAME = os.getenv("CQA_PROJECT_NAME")
    CQA_DEPLOYMENT_NAME = os.getenv("CQA_DEPLOYMENT_NAME")

    keyvault_url = os.getenv("KEYVAULT_URL")

    if not LANGUAGE_ENDPOINT or not keyvault_url:
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

    LANGUAGE_KEY = secret_client.get_secret("clu-key").value

    qa_client = QuestionAnsweringClient(
        endpoint=LANGUAGE_ENDPOINT,
        credential=AzureKeyCredential(LANGUAGE_KEY)
    )

    response = qa_client.get_answers(
        question=question,
        project_name=CQA_PROJECT_NAME,
        deployment_name=CQA_DEPLOYMENT_NAME,
        top=1
    )

    if not response.answers:
        return {
            "action": "faq",
            "answer": "Sorry, I don’t have an answer for that yet."
        }

    best = response.answers[0]

    return {
        "action": "faq",
        "answer": best.answer,
        "confidence": best.confidence
    }


# result = custom_qa_handler(question = 'What documents are required for insurance claims?')
# print(result['answer'])