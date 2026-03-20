import uuid
from datetime import datetime, timezone
from .search_client import get_search_client
import re



def safe_id(value: str) -> str:
    """
    Convert a string into a safe Azure AI Search document key.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def index_document_result(
    claim_id: str,
    source_file: str,
    doc_result: dict
):
    search_client = get_search_client()

    # Flatten text from pages
    all_text = []
    for page in doc_result.get("pages", []):
        all_text.extend(page.get("lines", []))

    # Include key phrases and entities (if present) in the searchable content
    key_phrases_list = doc_result.get("key_phrases", []) or []
    entities_list = doc_result.get("entities", []) or []

    all_text.extend(key_phrases_list)
    all_text.extend(entities_list)

    safe_file = safe_id(source_file)

    search_doc = {
        "id": f"{claim_id}-document-{safe_file}",
        "claim_id": claim_id,
        "source_type": "document",
        "source_file": source_file,
        "content": " ".join(all_text),
        # Store as arrays for Collection(Edm.String) fields in the index
        "key_phrases": key_phrases_list if key_phrases_list else [],
        "entities": entities_list if entities_list else [],
        "document_url": doc_result.get("document_url"),
        "created_at": utc_now()
    }

    print("Indexing document with created_at =", search_doc["created_at"])

    search_client.upload_documents([search_doc])


def index_image_result(
    claim_id: str,
    source_file: str,
    image_result: dict
):
    search_client = get_search_client()

    all_text = []

    # OCR text (stored as "content" in vision_processor)
    if image_result.get("content"):
        all_text.append(image_result["content"])

    # Caption
    if image_result.get("caption"):
        all_text.append(image_result["caption"])
    
    # Vehicle type and color for searchability
    if image_result.get("vehicle_type"):
        all_text.append(image_result["vehicle_type"])
    if image_result.get("color"):
        all_text.append(image_result["color"])

    safe_file = safe_id(source_file)

    search_doc = {
        "id": f"{claim_id}-image-{safe_file}",
        "claim_id": claim_id,
        "source_type": "image",
        "source_file": source_file,
        "content": " ".join(all_text),
        "caption": image_result.get("caption"),
        "image_url": image_result.get("image_url"),
        "vehicle_type": image_result.get("vehicle_type"),
        "color": image_result.get("color"),
        "damage_present": image_result.get("classification", {}).get("tag_name") == "Damaged" if image_result.get("classification") else False,
        "created_at": utc_now()
    }

    print("Indexing document with created_at =", search_doc["created_at"])

    search_client.upload_documents([search_doc])

