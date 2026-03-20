from agents.router_agent import router_agent
from agents.executor import execute_plan
import json

def test_agent_flow():
    query = "Show damaged cars from last week and summrize them and see if there is any fruads and generate a report"

    print("\n=== USER QUERY ===")
    print(query)

    # STEP 1: Router Agent → Execution Plan
    print("\n=== ROUTER AGENT OUTPUT (PLAN) ===")
    plan = router_agent(query)
    print(json.dumps(plan, indent=2))

    # STEP 2: Execute the Plan
    print("\n=== EXECUTOR OUTPUT ===")
    result = execute_plan(plan)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_agent_flow()
