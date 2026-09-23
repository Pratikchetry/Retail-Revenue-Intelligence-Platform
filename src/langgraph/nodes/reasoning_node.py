"""
Phase 1 — AI Retail Intelligence Platform
Reasoning Node.
"""

from src.agent.business_reasoning_agent import BusinessReasoningAgent
from src.utils.logger import get_logger
from src.langgraph.state import AgentState

log = get_logger(__name__)


def reasoning_node(state: AgentState) -> dict:
    """Synthesize a business answer from SQL result + context + history."""
    question = state["question"]
    context = state.get("context", "")
    rows = state.get("rows", "")
    history = state.get("history", [])  # Get conversation history

    log.info("Reasoning for: '%s'", question[:60])

    # Inject conversation history into the context so the LLM knows previous turns
    if history:
        history_str = "\n\nPREVIOUS CONVERSATION:\n"
        for msg in history[-4:]:  # Keep last 4 messages to save tokens
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            history_str += f"{role}: {content}\n"
        
        # Append history to the existing context
        context = context + history_str

    # Format the SQL result for the LLM
    sql_result_str = None
    if isinstance(rows, str) and rows in ("NO_SQL_REQUIRED", "INFORMATION_NOT_AVAILABLE"):
        # If SQL was skipped, pass None so the agent relies on the knowledge base
        sql_result_str = None
    elif hasattr(rows, "to_string"):
        # If it's a pandas DataFrame, format it nicely
        sql_result_str = rows.to_string(index=False)
    elif rows:
        sql_result_str = str(rows)

    reasoning_agent = BusinessReasoningAgent()
    result = reasoning_agent.reason(question, context, sql_result=sql_result_str)

    log.info("Reasoning answer: %s", result.answer[:100] if result.answer else "EMPTY")

    return {
        "answer": result.answer,
        "reasoning": result.reasoning,
        "evidence": result.evidence,
    }