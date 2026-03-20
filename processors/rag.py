from .search_client import get_search_client
import os




def retrieve_context(query: str, top_k=3):
    search_client = get_search_client()

    results = search_client.search(
        search_text=query,
        top=top_k
    )

    chunks = []
    for doc in results:
        if doc.get("content"):
            chunks.append(doc["content"])

    return "\n".join(chunks)

def build_rag_prompt(context: str, question: str):
    return f"""
You are an insurance assistant.

Answer the question using ONLY the information provided below.
If the answer is not present, say: "I do not have enough information to answer this question."

### Information:
{context}

### Question:
{question}

### Answer:
"""






