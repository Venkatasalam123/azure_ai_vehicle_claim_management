# agents/tools.py

from processors.router import search_handler
from processors.rag_explainer import generate_summary_from_docs
from processors.custom_question_answering import custom_qa_handler
from collections import Counter
from datetime import datetime

def search_agent(
    intent: str = "search",   # CLU intent: "search_image" | "search_document" | "search_mixed" | "search"
    source_type: str = None,   # "document" | "image" | None
    damage_present: bool = None,
    vehicle_type: str = None,
    date_range: dict = None,
    top_k: int = 10
):
    """
    Unified search agent for documents and images.
    Uses CLU intent if provided, otherwise defaults to "search".
    """
    print(f"\n=== SEARCH AGENT: INTENT RECEIVED ===")
    print(f"Intent: {intent}")
    print(f"Source Type: {source_type}")
    print(f"Damage Present: {damage_present}")
    print(f"Vehicle Type: {vehicle_type}")
    print(f"Date Range: {date_range}")
    print(f"Top K: {top_k}")

    entities = {}

    if source_type:
        entities["source_type"] = source_type   # <-- maps to index field

    if damage_present is not None:
        entities["damage_present"] = damage_present

    if vehicle_type:
        entities["vehicle_type"] = vehicle_type

    if date_range:
        entities["date_range"] = date_range

    return search_handler(
        intent=intent,   # Use CLU intent if provided
        entities=entities,
        top_k=top_k
    )



def explain_agent(documents):
    """
    Explain agent that summarizes documents.
    Requires search_agent output as input.
    """
    print(f"\n=== EXPLAIN AGENT: CALLED ===")
    docs = documents.get("results", [])
    print(f"Number of documents to summarize: {len(docs)}")
    return generate_summary_from_docs(docs)


def faq_agent(question: str):
    """
    FAQ agent that uses Azure Custom Question Answering service.
    Takes a question string directly.
    """
    print(f"\n=== FAQ AGENT: CALLED ===")
    print(f"Question: {question}")
    result = custom_qa_handler(question)
    print(f"FAQ Answer: {result.get('answer', 'N/A')}")
    print(f"FAQ Confidence: {result.get('confidence', 'N/A')}")
    return result


def fraud_agent(documents):
    """
    Stubbed fraud agent.
    Exists for completeness; not actively used yet.
    """

    return {
        "risk_level": "low",
        "signals": [],
        "note": "Fraud analysis not enabled for this request"
    }


def report_agent(summary, documents):
    """
    Stubbed report agent.
    Compiles a simple investigation report.
    """

    return {
        "report_text": f"""
CLAIM INVESTIGATION REPORT

SUMMARY:
{summary.get('summary')}

CONFIDENCE:
{summary.get('confidence')}

TOTAL DOCUMENTS ANALYZED:
{len(documents.get('results', []))}
"""
    }
