"""
Conversational chat bot processor with conversation history and context-aware responses.
Uses Azure OpenAI with RAG for relevant answers about insurance claims.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
from .rag import retrieve_context, build_rag_prompt


def get_openai_client():
    """Get Azure OpenAI client using Key Vault credentials."""
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

    return AzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2024-12-01-preview"
    )


def process_chat_message(
    user_message: str,
    conversation_history: list = None,
    use_rag: bool = True
) -> dict:
    """
    Process a chat message with conversation history and optional RAG context.
    
    Args:
        user_message: The user's message
        conversation_history: List of previous messages in format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        use_rag: Whether to use RAG for context-aware responses
    
    Returns:
        dict with 'response' and 'context_used' keys
    """
    if conversation_history is None:
        conversation_history = []
    
    # Check if this is a counting/aggregation question
    counting_keywords = ["how many", "count", "number of", "total", "how much"]
    is_counting_question = any(keyword in user_message.lower() for keyword in counting_keywords)
    
    # If it's a counting question about claims/documents/images, get count from search index
    if is_counting_question and ("claim" in user_message.lower() or "document" in user_message.lower() or "image" in user_message.lower() or "processed" in user_message.lower()):
        try:
            from .search_client import get_search_client
            search_client = get_search_client()
            
            # Get total count of all documents
            results = search_client.search(
                search_text="*",
                select=["id"],
                top=10000  # Get all documents to count
            )
            
            total_count = sum(1 for _ in results)
            
            # Count by type
            doc_results = search_client.search(
                search_text="*",
                filter="source_type eq 'document'",
                select=["id"],
                top=10000
            )
            doc_count = sum(1 for _ in doc_results)
            
            img_results = search_client.search(
                search_text="*",
                filter="source_type eq 'image'",
                select=["id"],
                top=10000
            )
            img_count = sum(1 for _ in img_results)
            
            # Build response with counts (plain text, no markdown)
            if total_count == 0:
                count_response = "There are no claims processed yet. The system is ready to process your first claim!"
            else:
                count_response = f"Based on the indexed data, there are {total_count} total claim(s) processed:\n• {doc_count} document(s)\n• {img_count} image(s)"
            
            return {
                "response": count_response,
                "context_used": True
            }
        except Exception as e:
            print(f"Error getting count from search index: {e}")
            # Fall through to normal processing if counting fails
    
    client = get_openai_client()
    
    # Build system message
    system_message = """You are a helpful insurance claims assistant. You help users with questions about:
- Insurance claim documents
- Vehicle damage assessments
- Claim processing procedures
- General insurance information

Be friendly, professional, and concise. If you don't have specific information, you can search the indexed documents for relevant details."""
    
    # Prepare messages for OpenAI
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history
    messages.extend(conversation_history)
    
    # Add current user message
    messages.append({"role": "user", "content": user_message})
    
    # If RAG is enabled, retrieve relevant context
    context_used = False
    if use_rag:
        try:
            context = retrieve_context(user_message, top_k=3)
            if context and context.strip():
                context_used = True
                # Modify the last user message to include context
                messages[-1]["content"] = f"""Context from insurance documents:
{context}

User question: {user_message}

Please answer the user's question using the context above if relevant. If the context doesn't contain relevant information, answer based on your general knowledge about insurance."""
        except Exception as e:
            print(f"Error retrieving context for chat: {e}")
            # Continue without context if RAG fails
    
    # Get response from OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        assistant_response = response.choices[0].message.content
        
        return {
            "response": assistant_response,
            "context_used": context_used
        }
    except Exception as e:
        return {
            "response": f"I apologize, but I encountered an error: {str(e)}. Please try again.",
            "context_used": False,
            "error": str(e)
        }
