import json
import uuid
from datetime import datetime
from typing import Any, Dict
import logging

logger = logging.getLogger("claim_ai")
logger.setLevel(logging.INFO)

def generate_trace_id() -> str:
    return str(uuid.uuid4())


def log_event(
    trace_id: str,
    stage: str,
    event: str,
    payload: Dict[str, Any]
):
    try:
        # log_record = {
        #     "trace_id": trace_id,
        #     "stage": stage,
        #     "event": event,
        #     "payload": payload,
        #     "timestamp": datetime.utcnow().isoformat() + "Z"
        # }

        # logger.info(json.dumps(log_record))
        logger.info(
            "claim_ai_event",
            extra={
                "custom_dimensions": {
                    "trace_id": trace_id,
                    "stage": stage,
                    "event": event,
                    "payload": payload
                }
            }
        )
    except Exception as e:
        print(f"ERROR in log_event: {e}", flush=True)
        print(f"trace_id: {trace_id}, stage: {stage}, event: {event}", flush=True)