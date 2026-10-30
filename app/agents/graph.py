"""Agent Graph - Compile nodes into a LangGraph state machine"""

import uuid
from typing import Optional

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from app.agents.state import AgentState
from app.agents.nodes.rewriter import rewriter_node
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node


def create_graph():
    """
    Create the LangGraph state machine.

    Flow:
    START → Rewriter (resolve follow-ups) → Planner (classify) → Retriever (search) → Responder (answer) → END

    The checkpointer keeps each chat's recent turns under its session ID, so
    the next query in that session starts with them in state.messages.

    Returns:
        Compiled LangGraph
    """
    # Create graph with AgentState as the state type
    graph = StateGraph(AgentState)

    # Add nodes (processing steps)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)

    # Add edges (define flow)
    # Rewriter always goes to planner
    graph.add_edge("rewriter", "planner")

    # Planner always goes to retriever
    graph.add_edge("planner", "retriever")

    # Retriever always goes to responder
    graph.add_edge("retriever", "responder")

    # Set entry point
    graph.set_entry_point("rewriter")

    # Set finish point
    graph.set_finish_point("responder")

    # Compile the graph - memory lives in this process and resets when the API restarts
    compiled_graph = graph.compile(checkpointer=InMemorySaver())

    print("✅ Agent graph created successfully!")
    print("   Flow: Rewriter → Planner → Retriever → Responder")

    return compiled_graph


def run_agent(query: str, graph, session_id: Optional[str] = None) -> dict:
    """
    Run the agent on a query.

    Args:
        query: User's question
        graph: Compiled LangGraph from create_graph()
        session_id: Chat session; queries with the same ID share memory.
                    Without one the query is answered on its own and nothing is remembered.

    Returns:
        Final agent state as a dict (LangGraph returns state fields as a dict)
    """
    # Every field starts fresh except messages, which the checkpointer restores
    initial_state = AgentState(query=query)
    thread_id = session_id or f"oneoff_{uuid.uuid4().hex}"

    # Run the graph
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    try:
        return graph.invoke(initial_state, {"configurable": {"thread_id": thread_id}})
    finally:
        if not session_id:
            graph.checkpointer.delete_thread(thread_id)
