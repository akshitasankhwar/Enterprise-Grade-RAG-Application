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


def run_agent(query: str, graph=None) -> AgentState:
    """
    Run the agent on a query.
    
    Args:
        query: User's question
        graph: Compiled LangGraph (creates if None)
        
    Returns:
        Final agent state with response
    """
    if graph is None:
        graph = create_graph()
    
    # Create initial state
    initial_state = AgentState(query=query)
    
    # Run the graph
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    final_state = graph.invoke(initial_state)
    print(type(final_state))
    print(final_state)
    
    return final_state


if __name__ == "__main__":
    # Test the graph
    from app.config import settings
    
    # Create graph
    print("Creating agent graph...")
    graph = create_graph()
    
    # Test 1: Conversational
    print("\n" + "="*60)
    print("TEST 1: Conversational Query")
    print("="*60)
    result1 = run_agent("Hello! How are you today?", graph)
    print(f"\nResponse: {result1.response}")
    print(f"Reasoning: {result1.reasoning}")
    
    # Test 2: Technical (if documents exist)
    print("\n" + "="*60)
    print("TEST 2: Technical Query")
    print("="*60)
    result2 = run_agent("What is machine learning?", graph)
    print(f"\nResponse: {result2.response}")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Documents used: {len(result2.reranked_docs)}")