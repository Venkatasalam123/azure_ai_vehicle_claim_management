from agents.tools import (
    search_agent,
    explain_agent,
    fraud_agent,
    report_agent,
    faq_agent
)

AGENT_REGISTRY = {
    "search_agent": search_agent,
    "explain_agent": explain_agent,
    "faq_agent": faq_agent
    # fraud_agent and report_agent removed from workflow
}


def resolve_refs(data, context):
    if isinstance(data, str) and data.startswith("$"):
        return context[data[1:]]

    if isinstance(data, dict):
        return {k: resolve_refs(v, context) for k, v in data.items()}

    return data


def execute_plan(plan: dict):
    context = {}

    for step in plan["steps"]:
        agent_name = step["agent"]
        raw_input = step["input"]

        resolved_input = resolve_refs(raw_input, context)
        tool_fn = AGENT_REGISTRY[agent_name]

        # 🔐 FILTER INPUT TO MATCH FUNCTION SIGNATURE
        allowed_args = tool_fn.__code__.co_varnames[:tool_fn.__code__.co_argcount]
        filtered_input = {
            k: v for k, v in resolved_input.items() if k in allowed_args
        }

        output = tool_fn(**filtered_input)

        context[f"{agent_name}_output"] = output

    return context