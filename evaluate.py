from evaluation_cases import EVALUATION_CASES
from agents.claim_process_agents import process_claim


def run_evaluations():
    results = []

    for case in EVALUATION_CASES:
        print(f"\n=== Running case: {case['name']} ===")

        output = process_claim(
            extracted_claim_data=case["input"]["claim_data"],
            vision_signals=case["input"]["vision_signals"],
            bill_data=case["input"]["bill_data"]
        )

        print("Output:", output)

        actual_status = output["final_decision"]["status"]
        expected_status = case["expected_status"]

        passed = actual_status == expected_status

        results.append({
            "case": case["name"],
            "expected": expected_status,
            "actual": actual_status,
            "passed": passed
        })

        print(f"Expected: {expected_status}")
        print(f"Actual:   {actual_status}")
        print("Result:   ", "PASS ✅" if passed else "FAIL ❌")

    return results


if __name__ == "__main__":
    run_evaluations()