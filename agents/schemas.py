from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ClaimUnderstandingOutput(BaseModel):
    normalized_claim: Dict[str, Any]
    missing_fields: List[str]
    issues: List[str]


class EvidenceConfidenceOutput(BaseModel):
    evidence_strength: str
    confidence_score: float
    notes: List[str]

class TriageOutput(BaseModel):
    status: str
    reason: str