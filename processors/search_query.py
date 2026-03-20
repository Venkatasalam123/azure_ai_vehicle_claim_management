from .search_client import get_search_client
from datetime import datetime, timedelta, timezone
import calendar
import re


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20
}

def normalize_number_words(text: str):
    for word, num in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(num), text)
    return text


def resolve_date_range(date_text: str):
    now = datetime.now(timezone.utc)

    # Normalize text
    text = date_text.lower().strip()
    text = normalize_number_words(text)

    # TODAY
    if text == "today":
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return start.isoformat(), now.isoformat()

    # YESTERDAY
    if text == "yesterday":
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=1)
        end = start + timedelta(hours=23, minutes=59, seconds=59)
        return start.isoformat(), end.isoformat()

    # GENERIC RANGE
    pattern = r"(last|past)\s+(\d+)\s+(day|days|week|weeks|month|months)"
    match = re.search(pattern, text)

    if match:
        _, number, unit = match.groups()
        number = int(number)

        if "day" in unit:
            start = now - timedelta(days=number)
            return start.isoformat(), now.isoformat()

        if "week" in unit:
            start = now - timedelta(weeks=number)
            return start.isoformat(), now.isoformat()

        if "month" in unit:
            month = now.month - number
            year = now.year

            while month <= 0:
                month += 12
                year -= 1

            start = datetime(year, month, 1, tzinfo=timezone.utc)
            return start.isoformat(), now.isoformat()

    return None, None



def search_claims(
    query: str,
    top_k: int = 10,
    claim_id: str | None = None
) -> list:
    """
    Search indexed claim documents/images.
    """

    search_client = get_search_client()

    filter_expr = None
    if claim_id:
        filter_expr = f"claim_id eq '{claim_id}'"

    results = search_client.search(
        search_text=query,
        filter=filter_expr,
        query_type="semantic",
        semantic_configuration_name="claims-semantic-config",
        top=top_k
        )

    response = []

    for result in results:
        content = result.get("content", "")
        # Truncate content_preview to ~300 chars
        content_preview = content[:300] + "..." if len(content) > 300 else content
        
        response.append({
            "id": result.get("id"),
            "claim_id": result.get("claim_id"),
            "source_type": result.get("source_type"),
            "source_file": result.get("source_file"),
            "image_url": result.get("image_url"),
            "document_url": result.get("document_url"),
            "key_phrases": result.get("key_phrases"),
            "entities": result.get("entities"),
            "content_preview": content_preview,
            "created_at": result.get("created_at")
        })

    return response


def build_search_filter(intent, entities):

    
    filters = []

    # Source type based on intent
    if intent == "search_image":
        filters.append("source_type eq 'image'")
    elif intent == "search_document":
        filters.append("source_type eq 'document'")

    # Damage filter
    if entities.get("damage_present") is True:
        filters.append("damage_present eq true")

    # # Document type filter
    # if "document_type" in entities:
    #     filters.append(f"document_type eq '{entities['document_type']}'")

    # Date range
    if "date_range" in entities:
        print(entities["date_range"])
        start, end = resolve_date_range(entities["date_range"])
        print(start, end)
        if start and end:
            filters.append(
                f"created_at ge {start} and created_at le {end}"
            )

    return " and ".join(filters) if filters else None
