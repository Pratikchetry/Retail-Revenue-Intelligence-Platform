"""
Phase 2 — AI Retail Intelligence Platform
LangGraph StateGraph — the agent brain.
"""

import functools
from langgraph.graph import StateGraph, END

from src.langgraph.state import AgentState
from src.langgraph.nodes.router_node import router_node
from src.langgraph.nodes.rag_retriever_node import rag_retriever_node
from src.langgraph.nodes.context_compressor_node import compress_context_node
from src.langgraph.nodes.sql_generator_node import sql_generator_node
from src.langgraph.nodes.sql_validator_node import sql_validator_node
from src.langgraph.nodes.sql_executor_node import sql_executor_node
from src.langgraph.nodes.reasoning_node import reasoning_node
from src.langgraph.nodes.recommendation_node import recommendation_node
from src.langgraph.nodes.critic_node import critic_node
from src.langgraph.nodes.metadata_verifier_node import metadata_verifier_node
from src.langgraph.nodes.forecast_node import forecast_node
from src.langgraph.nodes.anomaly_node import anomaly_node
from src.utils.logger import get_logger

log = get_logger(__name__)


def decline_node(state: AgentState) -> dict:
    """Graceful, honest decline for OUT_OF_SCOPE questions."""
    return {
        "answer": (
            "This question is outside the scope of the retail revenue "
            "warehouse. I can answer questions about sales, customers, "
            "products, revenue, forecasts, and anomalies in our data."
        ),
        "recommendation": "",
        "evidence": "No data exists for this topic in the warehouse.",
        "critic_score": 1.0,
        "critic_passes": True,
    }


# ------------------------------------------------------------------
# Routing functions
# ------------------------------------------------------------------

def _route_after_router(state) -> str:
    route = state.get("route", "SQL_LOOKUP")
    if route == "OUT_OF_SCOPE":
        return "decline"
    if route == "FORECAST":
        return "forecast"
    if route == "ANOMALY":
        return "anomaly"
    return "rag"


def _route_after_validator(state) -> str:
    if not state.get("is_valid", False) and state.get("attempt", 0) < state.get("max_attempts", 3):
        return "retry"
    return "exec"


def _route_after_critic(state) -> str:
    if not state.get("critic_passes", True) and state.get("attempt", 0) < state.get("max_attempts", 3):
        return "retry"
    return "done"


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph():
    """Compile and return the full agent StateGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_retriever_node)
    workflow.add_node("compress", compress_context_node)
    workflow.add_node("sql_gen", sql_generator_node)
    workflow.add_node("sql_val", sql_validator_node)
    workflow.add_node("sql_exec", sql_executor_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("verifier", metadata_verifier_node)
    workflow.add_node("recommendation", recommendation_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("decline", decline_node)
    workflow.add_node("forecast", forecast_node)
    workflow.add_node("anomaly", anomaly_node)

    # Entry
    workflow.set_entry_point("router")

    # Router dispatch
    workflow.add_conditional_edges("router", _route_after_router, {
        "decline": "decline",
        "forecast": "forecast",
        "anomaly": "anomaly",
        "rag": "rag",
    })

    # SQL chain
    workflow.add_edge("rag", "compress")
    workflow.add_edge("compress", "sql_gen")
    workflow.add_edge("sql_gen", "sql_val")
    workflow.add_conditional_edges("sql_val", _route_after_validator, {
        "retry": "sql_gen",
        "exec": "sql_exec",
    })
    workflow.add_edge("sql_exec", "reasoning")

    # Reasoning → verifier (cross-check) → recommendation → critic
    workflow.add_edge("reasoning", "verifier")
    workflow.add_edge("verifier", "recommendation")
    workflow.add_edge("recommendation", "critic")
    workflow.add_conditional_edges("critic", _route_after_critic, {
        "retry": "sql_gen",
        "done": END,
    })

    # Forecast + Anomaly routes flow into recommendation → critic too
    workflow.add_edge("forecast", "recommendation")
    workflow.add_edge("anomaly", "recommendation")

    # Terminal
    workflow.add_edge("decline", END)

    return workflow.compile()


# Cache the compiled graph so we only build it ONCE at startup.
@functools.lru_cache(maxsize=1)
def get_compiled_graph():
    log.info("Compiling LangGraph agent for the first time...")
    graph = build_graph()
    log.info("LangGraph agent compiled and cached successfully.")
    return graph


def run_agent(question: str, history: list = None) -> dict:
    """One-call entry point: ask a question, get the full agent result."""
    graph = get_compiled_graph()
    log.info("Running agent for: '%s'", question[:60])
    
    # Initialize state with conversation history
    initial_state = {
        "question": question,
        "history": history or [],
        "max_attempts": 3,
        "attempt": 0
    }
    
    result = graph.invoke(initial_state)
    log.info("Agent complete | critic_score=%.2f", result.get("critic_score", 0))
    return result