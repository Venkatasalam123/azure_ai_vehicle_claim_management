from datetime import datetime
import json
import time
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential
from azure.ai.agents import AgentsClient
from agents.logger import log_event, generate_trace_id
from agents.schemas import (
    ClaimUnderstandingOutput,
    EvidenceConfidenceOutput,
    TriageOutput
)
from pydantic import ValidationError

PROJECT_ENDPOINT = "https://aih-insclaims-dev.services.ai.azure.com/api/projects/proj-insclaims"

CLAIM_UNDERSTANDING_AGENT_ID = "asst_jjG7PjuICP4xxiX9dmC0Z5mH"
EVIDENCE_CONFIDENCE_AGENT_ID = "asst_C8aYGQMEBL61mJj6GQvqKZFe"
CLAIM_TRIAGE_AGENT_ID = "asst_5F2rvi3sJO68qTIuUDIYob08"

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

credential = ClientSecretCredential(
    tenant_id=os.getenv("AZURE_TENANT_ID"),
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
)

agent_client = AgentsClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential
)


def run_agent(agent_id: str, payload: dict, trace_id: str, stage: str, schema) -> dict:
    """
    Runs a Foundry agent using AgentsClient and returns JSON output.
    """
    
    log_event(
        trace_id=trace_id,
        stage=stage,
        event="agent_input",
        payload=payload
    )

    # 1. Create thread
    thread = agent_client.threads.create()

    # 2. Send input
    agent_client.messages.create(
        thread_id=thread.id,
        role="user",
        content=json.dumps(payload)
    )

    # 3. Run agent
    run = agent_client.runs.create(
        thread_id=thread.id,
        agent_id=agent_id
    )

    # 4. Wait for completion
    while run.status in ("queued", "in_progress"):
        time.sleep(1)
        run = agent_client.runs.get(
            thread_id=thread.id,
            run_id=run.id
        )

    if run.status != "completed":
        raise RuntimeError(f"Agent run failed with status: {run.status}")

    # 5. Read response
    messages = agent_client.messages.list(thread_id=thread.id)

    for msg in messages:
        if msg.role == "assistant":
            raw_output = json.loads(msg.content[0].text.value)

            log_event(trace_id, stage, "agent_output_raw", raw_output)

            try:
                validated = schema(**raw_output)
            except ValidationError as e:
                log_event(trace_id, stage, "schema_validation_error", e.errors())
                raise

            log_event(trace_id, stage, "agent_output_validated", validated.dict())

            return validated.dict()

    raise RuntimeError("No agent response found")


def process_claim(
    extracted_claim_data: dict,
    vision_signals: dict,
    bill_data: dict | None
) -> dict:
    """
    End-to-end claim processing using 3 Foundry agents.
    """

    trace_id = generate_trace_id()

    log_event(
        trace_id=trace_id,
        stage="claim_submission",
        event="received",
        payload={
            "claim_data": extracted_claim_data,
            "vision_signals": vision_signals,
            "bill_data": bill_data
        }
    )

    # -------------------------
    # 1️⃣ Claim Understanding
    # -------------------------
    from datetime import datetime
    current_date = datetime.utcnow().date().isoformat()
    claim_understanding_input = {
        **extracted_claim_data,
        **(bill_data or {}),
        "current_date": current_date
    }
    
    # Map repair_date to service_date if service_date doesn't exist but repair_date does
    if 'service_date' not in claim_understanding_input and 'repair_date' in claim_understanding_input:
        claim_understanding_input['service_date'] = claim_understanding_input['repair_date']

    claim_understanding = run_agent(
        CLAIM_UNDERSTANDING_AGENT_ID,
        claim_understanding_input,
        trace_id=trace_id,
        stage="claim_understanding",
        schema=ClaimUnderstandingOutput
    )

    print("=== CLAIM UNDERSTANDING OUTPUT ===")
    print(json.dumps(claim_understanding, indent=2))

    # -------------------------
    # 2️⃣ Evidence Confidence
    # -------------------------
    evidence_confidence_input = {
        **vision_signals,
        "missing_documents": claim_understanding["missing_fields"]
    }

    evidence_confidence = run_agent(
        EVIDENCE_CONFIDENCE_AGENT_ID,
        evidence_confidence_input,
        trace_id=trace_id,
        stage="evidence_confidence",
        schema=EvidenceConfidenceOutput
    )

    print("=== EVIDENCE CONFIDENCE OUTPUT ===")
    print(json.dumps(evidence_confidence, indent=2))

    # -------------------------
    # 3️⃣ Claim Intake Triage
    # -------------------------
    triage_input = {
        "claim_amount": claim_understanding["normalized_claim"].get("claim_amount"),
        "evidence_strength": evidence_confidence["evidence_strength"],
        "confidence_score": evidence_confidence["confidence_score"],
        "missing_fields": claim_understanding["missing_fields"],
        "issues": claim_understanding["issues"],
        "damage_present": vision_signals.get("damage_present", False)
    }

    triage_decision = run_agent(
        CLAIM_TRIAGE_AGENT_ID,
        triage_input,
        trace_id=trace_id,
        stage="claim_triage",
        schema=TriageOutput
    )

    print("=== CLAIM INTAKE TRIAGE OUTPUT ===")
    print(json.dumps(triage_decision, indent=2))

    return {
        "claim_understanding": claim_understanding,
        "evidence_confidence": evidence_confidence,
        "final_decision": triage_decision
    }


def main():
    """
    Simulates UI flow:
    - User uploads damage image
    - User uploads repair bill (optional)
    - User clicks OK
    """

    # ---- Simulated extraction outputs ----
    extracted_claim_data = {
        "claim_amount": "4.5L",
        "vehicle_type": "car"
    }

    vision_signals = {
        "vision_confidence": 0.72,
        "damage_severity": "high",
        "semantic_alignment": True,
        "damage_present": True
    }

    bill_data = {
        "repair_amount": "380000",
        "repair_date": "2026-02-11"
    }
    # bill_data = None  # ← try this to test image-only flow

    # ---- Process claim ----
    result = process_claim(
        extracted_claim_data=extracted_claim_data,
        vision_signals=vision_signals,
        bill_data=bill_data
    )

    print("\n=== FINAL CLAIM RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()