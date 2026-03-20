import os
import re
from dotenv import load_dotenv
from pathlib import Path

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def serialize_image_point(point):
    """
    Convert ImagePoint to JSON-serializable dict.
    """
    if point is None:
        return None
    return {
        "x": point.x,
        "y": point.y
    }

VEHICLE_TYPES = {"car", "truck", "bus", "motorcycle", "bike"}
COLORS = {
    # Primary
    "black", "white", "red", "blue", "green", "yellow",
    "orange", "brown", "purple", "pink",

    # Neutral / automotive
    "gray", "grey", "silver", "beige", "cream", "ivory", "tan",

    # Dark / metallic
    "maroon", "navy", "charcoal", "teal", "bronze", "gold"
}

def analyze_claim_image(image_path: str) -> dict:
    """
    Analyze an insurance claim image and extract ONLY
    what is required for indexing and filtering.
    """

    # Load env
    env_path = Path(__file__).parent / '.env'
    load_dotenv(dotenv_path=env_path)

    vision_endpoint = os.getenv("VISION_ENDPOINT")
    keyvault_url = os.getenv("KEYVAULT_URL")

    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    secret_client = SecretClient(vault_url=keyvault_url, credential=credential)
    vision_key = secret_client.get_secret("computervision-key").value

    vision_client = ImageAnalysisClient(
        endpoint=vision_endpoint,
        credential=AzureKeyCredential(vision_key)
    )

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = vision_client.analyze(
        image_data=image_bytes,
        visual_features=[
            VisualFeatures.READ,
            VisualFeatures.TAGS,
            VisualFeatures.CAPTION, 
            VisualFeatures.OBJECTS
        ]
    )

    print(response.tags)
    print(response.read)
    print(response.caption)
    print(response.objects)

    # ---- Extract OCR (flattened) ----
    ocr_text = []
    if response.read and response.read.blocks:
        for block in response.read.blocks:
            for line in block.lines or []:
                ocr_text.append(line.text)

    content_text = " ".join(ocr_text)

    # ---- Extract tags ----
    tags = []
    if response.tags:
        for t in response.tags:
            # Handle both string tags and tag objects with .name attribute
            if isinstance(t, str):
                tags.append(t.lower())
            elif hasattr(t, 'name'):
                tags.append(t.name.lower())
            else:
                # Fallback: try to convert to string
                tags.append(str(t).lower())

    # ---- Caption ----
    caption = response.caption.text if response.caption else None

    # ---- Derive vehicle type and color from caption ----
    vehicle_type = None
    color = None
    if caption:
        caption_lower = caption.lower()
        # Split caption into words and strip punctuation
        caption_words = [re.sub(r'[^\w]', '', word) for word in caption_lower.split()]
        
        # Find vehicle type in caption
        for word in caption_words:
            if word in VEHICLE_TYPES:
                vehicle_type = word
                break
        
        # Find color in caption
        for word in caption_words:
            if word in COLORS:
                color = word
                break

    extracted_data = {
        "caption": caption,
        "content": content_text,
        "tags": tags,
        "vehicle_type": vehicle_type,
        "color": color
    }

    print(extracted_data)

    return extracted_data
