from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient


def classify_claim_image(image_path: str) -> dict:
    """
    Classify an insurance claim image using Custom Vision.
    Returns structured data with predictions and best prediction.
    """
    
    # Load environment variables from parent directory
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)

    # -----------------------------
    # Read config (NON-SECRET)
    # -----------------------------
    endpoint = os.getenv("CUSTOM_VISION_PREDICTION_ENDPOINT")
    projectid = os.getenv("CUSTOM_VISION_PROJECT_ID")
    model_name = os.getenv("CUSTOM_VISION_MODEL_NAME")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url or not projectid or not model_name:
        raise ValueError(
            "CUSTOM_VISION_PREDICTION_ENDPOINT, KEYVAULT_URL, CUSTOM_VISION_PROJECT_ID, or CUSTOM_VISION_MODEL_NAME not set"
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
    # Fetch Custom Vision Key
    # -----------------------------
    key = secret_client.get_secret(
        "customvision-key"
    ).value

    # -----------------------------
    # Create Custom Vision Client
    # -----------------------------
    client = CustomVisionPredictionClient(
        endpoint=endpoint,
        credentials=ApiKeyCredentials(in_headers={"Prediction-key": key})
    )

    # -----------------------------
    # Read Image
    # -----------------------------
    with open(image_path, "rb") as f:
        image_data = f.read()

    # -----------------------------
    # Classify Image
    # -----------------------------
    response = client.classify_image(projectid, model_name, image_data)

    # -----------------------------
    # Extract Predictions
    # -----------------------------
    predictions = []
    for pred in response.predictions:
        predictions.append({
            "tag_name": pred.tag_name,
            "probability": pred.probability
        })

    # Get best prediction
    best_prediction = None
    if predictions:
        best_prediction = max(predictions, key=lambda x: x['probability'])

    classification_data = {
        "predictions": predictions,
        "best_prediction": best_prediction,
        "tag_name": best_prediction['tag_name'] if best_prediction else None,
        "confidence": best_prediction['probability'] if best_prediction else None
    }

    return classification_data