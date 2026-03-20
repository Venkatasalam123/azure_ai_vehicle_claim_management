import os
import re
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import numpy as np

from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


DAMAGE_KEYWORDS = {
    "damage", "damaged", "crash", "accident", "collision",
    "dent", "broken", "smashed", "scratched", "wrecked"
}


def detect_metadata_present(image_path: str) -> bool:
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        return exif is not None and len(exif) > 0
    except Exception:
        return False


def detect_blur(image_path: str, threshold: float = 100.0) -> bool:
    """
    Simple blur detection using Laplacian variance.
    Low variance → blurry image.
    """
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    variance = np.var(np.gradient(img_array))
    return bool(variance < threshold)

def detect_texture_anomaly(tags: list, caption_confidence: float) -> bool:
    """
    Heuristic: AI images often have low semantic confidence
    but visually 'perfect' scenes.
    """
    return caption_confidence < 0.35 and len(tags) < 5


def assess_damage_consistency(vehicle_detected: bool, damage_keywords_present: bool) -> str:
    if vehicle_detected and not damage_keywords_present:
        return "low"
    if vehicle_detected and damage_keywords_present:
        return "high"
    return "unknown"


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


def make_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types
    to native Python types.
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

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
    env_path = Path(__file__).parent.parent.parent / ".env"
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

    # ---- Extract objects ----
    objects = []
    object_confidences = []

    if response.objects:
        for obj in response.objects:
            if hasattr(obj, "tags") and obj.tags:
                for tag in obj.tags:
                    objects.append(tag.name.lower())
                    object_confidences.append(tag.confidence)

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

    # ---- Authenticity Signals ----

    # Is vehicle detected?
    vehicle_detected = vehicle_type is not None or any(
        v in objects for v in VEHICLE_TYPES
    )

    # Is damage mentioned?
    damage_keywords_present = False
    combined_text = " ".join(tags + ocr_text)
    for word in DAMAGE_KEYWORDS:
        if word in combined_text.lower():
            damage_keywords_present = True
            break

    # Is text present in image? (many AI images avoid text)
    text_present = len(ocr_text) > 0

    # Caption confidence proxy
    caption_confidence = response.caption.confidence if response.caption else 0.0

    # Tag confidence average (only for tag objects with confidence attribute)
    tag_confidences = [
        t.confidence for t in response.tags 
        if hasattr(t, 'confidence') and not isinstance(t, str)
    ]
    avg_tag_confidence = (
        sum(tag_confidences) / len(tag_confidences)
        if tag_confidences else 0.0
    )

    # Heuristic suspicious scene check
    suspicious_scene = (
        caption_confidence < 0.3 and
        avg_tag_confidence < 0.5 and
        vehicle_detected
    )

    # Overall vision confidence
    vision_confidence = round(
        max(caption_confidence, avg_tag_confidence),
        2
    )

    # Additional authenticity checks
    metadata_present = detect_metadata_present(image_path)
    blur_detected = detect_blur(image_path)
    texture_anomaly = detect_texture_anomaly(tags, caption_confidence)

    damage_consistency = assess_damage_consistency(
        vehicle_detected=vehicle_detected,
        damage_keywords_present=damage_keywords_present
    )

    extracted_data = {
        # Existing fields (keep them)
        "caption": caption,
        "content": content_text,
        "tags": tags,
        "vehicle_type": vehicle_type,
        "color": color,

        # ---- Image authenticity signals ----
        "authenticity_signals": {
            "vehicle_detected": vehicle_detected,
            "damage_keywords_present": damage_keywords_present,
            "text_present": text_present,
            "caption_confidence": caption_confidence,
            "avg_tag_confidence": avg_tag_confidence,
            "suspicious_scene": suspicious_scene,
            "vision_confidence": vision_confidence,
            "blur_detected": blur_detected,
            "metadata_present": metadata_present,
            "damage_consistency": damage_consistency,
            "texture_anomaly": texture_anomaly
        }
    }

    print(extracted_data)

    # Ensure all values are JSON serializable
    extracted_data = make_json_serializable(extracted_data)

    return extracted_data
