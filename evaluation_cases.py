EVALUATION_CASES = [
    {
        "name": "Low value claim with strong evidence",
        "input": {
            "claim_data": {"claim_amount": 4700},
            "vision_signals": {
                "damage_present": True,
                "evidence_strength": "HIGH",
                "confidence_score": 0.9
            },
            "bill_data": {
                "repair_amount": 4700,
                "repair_date": "2026-02-11"
            }
        },
        "expected_status": "PROCESSED"
    },
    {
        "name": "High value claim requires manual review",
        "input": {
            "claim_data": {"claim_amount": 450000},
            "vision_signals": {
                "damage_present": True,
                "evidence_strength": "HIGH",
                "confidence_score": 0.85
            },
            "bill_data": {
                "repair_amount": 450000,
                "repair_date": "2026-02-11"
            }
        },
        "expected_status": "WAITING"
    },
    {
        "name": "No damage present should be rejected",
        "input": {
            "claim_data": {"claim_amount": 5000},
            "vision_signals": {
                "damage_present": False,
                "evidence_strength": "LOW",
                "confidence_score": 0.3
            },
            "bill_data": {
                "repair_amount": 5000,
                "repair_date": "2026-02-11"
            }
        },
        "expected_status": "REJECTED"
    },
    {
        "name": "Missing bill information",
        "input": {
            "claim_data": {"claim_amount": 8000},
            "vision_signals": {
                "damage_present": True,
                "evidence_strength": "MEDIUM",
                "confidence_score": 0.6
            },
            "bill_data": None
        },
        "expected_status": "WAITING"
    }
]