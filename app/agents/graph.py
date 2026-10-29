"""Agent Graph - Compile nodes into a LangGraph state machine"""

from langgraph.graph import StateGraph
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.responder import responder_node


def create_graph():
    """
    Create the LangGraph state machine.
    
    Flow:
    START → Planner (classify) → Retriever (search) → Responder (answer) → END
    
    Returns:
        Compiled LangGraph
    """
    # Create graph with AgentState as the state type
    graph = StateGraph(AgentState)
    
    # Add nodes (processing steps)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("responder", responder_node)
    
    # Add edges (define flow)
    # Planner always goes to retriever
    graph.add_edge("planner", "retriever")
    
    # Retriever always goes to responder
    graph.add_edge("retriever", "responder")
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Set finish point
    graph.set_finish_point("responder")
    
    # Compile the graph
    compiled_graph = graph.compile()
    
    print("✅ Agent graph created successfully!")
    print("   Flow: Planner → Retriever → Responder")
    
    return compiled_graph


def run_agent(query: str, graph) -> dict:
    """
    Run the agent on a query.

    Args:
        query: User's question
        graph: Compiled LangGraph from create_graph()

    Returns:
        Final agent state as a dict (LangGraph returns state fields as a dict)
    """
    # Create initial state
    initial_state = AgentState(query=query)
    
    # Run the graph
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    return graph.invoke(initial_state)
