import json
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI


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

client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-02-15-preview",
    azure_endpoint=OPENAI_ENDPOINT
)


def router_agent(user_query: str, clu_result: dict = None):
    system_prompt = """
You are a planning agent for a vehicle insurance claims system.

Your ONLY job is to create an execution plan in JSON.
You do NOT answer the user.
You do NOT explain your reasoning.
You do NOT include any text outside JSON.

You must choose from the following agents ONLY:
- search_agent
- explain_agent
- faq_agent

Rules:
1. Output MUST be valid JSON.
2. Output MUST contain a top-level key "steps".
3. Each step MUST have:
   - "agent"
   - "input"
4. Never invent data.
5. Never hallucinate results.
6. If CLU intent is "faq" → use faq_agent directly with the user query (NO search_agent needed).
7. If CLU intent is "explain" → use search_agent FIRST, then explain_agent with search results.
8. Use search_agent whenever information must be retrieved (except for FAQ queries).
9. Use explain_agent ONLY after search_agent.
10. Do NOT skip search_agent for investigation or summary tasks (unless intent is "faq").
11. Keep plans minimal and logical.
12. NEVER include extra keys in agent inputs.
13. explain_agent input must contain ONLY "documents".
14. faq_agent input must contain ONLY "question" (the user's query text).
15. If CLU intent is provided, use it in search_agent input as "intent" field.
16. search_agent input may include: intent, source_type, damage_present, vehicle_type, date_range, top_k
"""

    clu_info = ""
    if clu_result:
        intent = clu_result.get("intent", "")
        entities = clu_result.get("entities", {})
        confidence = clu_result.get("confidence", 0)
        print(f"\n=== ROUTER AGENT: CLU RESULT RECEIVED ===")
        print(f"CLU Intent: {intent}")
        print(f"CLU Confidence: {confidence:.2%}")
        print(f"CLU Entities: {json.dumps(entities, indent=2) if entities else 'None'}")
        clu_info = f"""

CLU Analysis Result:
- Intent: {intent} (confidence: {confidence:.2%})
- Entities: {json.dumps(entities, indent=2) if entities else "None"}

IMPORTANT: Use the CLU intent in search_agent input. The intent should be passed as "intent" field.
"""
    else:
        print(f"\n=== ROUTER AGENT: NO CLU RESULT PROVIDED ===")

    user_prompt = f"""
User query:
"{user_query}"
{clu_info}

Interpret the intent and produce an execution plan.

Guidelines:
- If CLU intent is "faq" → use faq_agent directly with the user query (NO search needed).
- If CLU intent is "explain" → use search_agent FIRST, then explain_agent with search results.
- If the user asks to FIND, SHOW, LIST, FILTER → use search_agent.
- If the user asks to SUMMARIZE, EXPLAIN, ANALYZE → use explain_agent AFTER search_agent.
- If CLU intent is provided, include it in search_agent input (for search intents).

Example output formats:

For FAQ intent:
{{
  "steps": [
    {{
      "agent": "faq_agent",
      "input": {{
        "question": "What documents are required for insurance claims?"
      }}
    }}
  ]
}}

For Explain intent:
{{
  "steps": [
    {{
      "agent": "search_agent",
      "input": {{
        "intent": "search_mixed"
      }}
    }},
    {{
      "agent": "explain_agent",
      "input": {{
        "documents": "$search_agent_output"
      }}
    }}
  ]
}}

For Search intent:
{{
  "steps": [
    {{
      "agent": "search_agent",
      "input": {{
        "intent": "search_image",
        "damage_present": true,
        "vehicle_type": "car",
        "date_range": "last_week"
      }}
    }}
  ]
}}

For complex queries with search and explain:
{{
  "steps": [
    {{
      "agent": "search_agent",
      "input": {{
        "intent": "search_mixed"
      }}
    }},
    {{
      "agent": "explain_agent",
      "input": {{
        "documents": "$search_agent_output"
      }}
    }}
  ]
}}

Now produce the execution plan.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)
