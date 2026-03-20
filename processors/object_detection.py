from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient


def detect_claim_damage(image_path: str) -> dict:
    """
    Detect damaged areas in an insurance claim image using Custom Vision Object Detection.
    Returns all detections and the highest-confidence bounding box.
    """

    # Load environment variables from parent directory
    BASE_DIR = Path(__file__).resolve().parents[2]
    load_dotenv(BASE_DIR / ".env")

    # ---------------------------------
    # Read config
    # ---------------------------------
    endpoint = os.getenv("CUSTOM_VISION_PREDICTION_ENDPOINT")
    projectid = os.getenv("CUSTOM_VISION_OBJECT_DETECTION_PROJECT_ID")
    model_name = os.getenv("CUSTOM_VISION_OBJECT_DETECTION_MODEL_NAME")
    keyvault_url = os.getenv("KEYVAULT_URL")

    if not endpoint or not keyvault_url or not projectid or not model_name:
        raise ValueError("Required environment variables not set")

    # ---------------------------------
    # Authenticate to Key Vault
    # ---------------------------------
    credential = ClientSecretCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID"),
        client_secret=os.getenv("AZURE_CLIENT_SECRET")
    )

    secret_client = SecretClient(
        vault_url=keyvault_url,
        credential=credential
    )

    # ---------------------------------
    # Fetch Custom Vision Key
    # ---------------------------------
    prediction_key = secret_client.get_secret("customvision-key").value

    client = CustomVisionPredictionClient(
        endpoint=endpoint,
        credentials=ApiKeyCredentials(
            in_headers={"Prediction-key": prediction_key}
        )
    )

    # ---------------------------------
    # Read image
    # ---------------------------------
    with open(image_path, "rb") as f:
        image_data = f.read()

    # ---------------------------------
    # Detect objects
    # ---------------------------------
    response = client.detect_image(projectid, model_name, image_data)

    detections = []

    for pred in response.predictions:
        detections.append({
            "tag_name": pred.tag_name,
            "probability": pred.probability,
            "bounding_box": {
                "left": pred.bounding_box.left,
                "top": pred.bounding_box.top,
                "width": pred.bounding_box.width,
                "height": pred.bounding_box.height
            }
        })

    # ---------------------------------
    # Get highest confidence detection
    # ---------------------------------
    best_detection = None
    if detections:
        best_detection = max(detections, key=lambda x: x["probability"])

    return {
        "detections": detections,
        "best_detection": best_detection,
        "tag_name": best_detection["tag_name"] if best_detection else None,
        "confidence": best_detection["probability"] if best_detection else None,
        "bounding_box": best_detection["bounding_box"] if best_detection else None
    }


from PIL import Image, ImageDraw

def show_image(image_path):
    img = Image.open(image_path)
    img.show()   # opens default image viewer

def to_pixel_bbox(bbox, image_path):
    img = Image.open(image_path)
    img_w, img_h = img.size

    return {
        "x1": int(bbox["left"] * img_w),
        "y1": int(bbox["top"] * img_h),
        "x2": int((bbox["left"] + bbox["width"]) * img_w),
        "y2": int((bbox["top"] + bbox["height"]) * img_h)
    }


def draw_bbox(image_path, bbox_pixels):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    draw.rectangle(
        [bbox_pixels["x1"], bbox_pixels["y1"],
         bbox_pixels["x2"], bbox_pixels["y2"]],
        outline="red",
        width=3
    )

    img.show()

# image_path = Path(__file__).parent.parent / "sample_images" / "damage1.jpeg"
# result = detect_claim_damage(image_path)
# bbox_pixels = to_pixel_bbox(result["bounding_box"], image_path)
# draw_bbox(image_path, bbox_pixels)


