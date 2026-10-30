"""LangGraph topology, kept in the original execution order."""

from langgraph.graph import END, START, StateGraph

from ..state import AgentState


def build_graph(runtime):
    builder = StateGraph(AgentState)
    builder.add_node("load_context", runtime.load_context)
    builder.add_node("compact_context", runtime.compact_context)
    builder.add_node("route_and_extract", runtime.route_and_extract)
    builder.add_node("execute_tool", runtime.execute_tool)
    builder.add_node("build_response", runtime.build_response)
    builder.add_node("verify_grounding", runtime.verify_grounding)
    builder.add_node("verify_answer_faithfulness", runtime.verify_answer_faithfulness)
    builder.add_node("persist", runtime.persist)
    builder.add_edge(START, "load_context")
    builder.add_conditional_edges(
        "load_context",
        lambda state: (
            "compact_context"
            if (state.get("context_budget") or {}).get("needs_compaction")
            else "route_and_extract"
        ),
        {
            "compact_context": "compact_context",
            "route_and_extract": "route_and_extract",
        },
    )
    builder.add_edge("compact_context", "route_and_extract")
    builder.add_edge("route_and_extract", "execute_tool")
    builder.add_edge("execute_tool", "build_response")
    builder.add_edge("build_response", "verify_grounding")
    builder.add_edge("verify_grounding", "verify_answer_faithfulness")
    builder.add_edge("verify_answer_faithfulness", "persist")
    builder.add_edge("persist", END)
    return builder.compile()
