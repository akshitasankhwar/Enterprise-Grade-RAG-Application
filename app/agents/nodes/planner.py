"""Planner Node - Classify the query and decide routing"""
from app.config import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.agents.state import AgentState


def planner_node(state: AgentState) -> AgentState:
    """
    Planner Node: Classify whether this is a conversational or technical query.
    
    This helps decide:
    - CONVERSATIONAL: Greeting, small talk → Respond directly
    - TECHNICAL: Asking about documents → Need to retrieve and search
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with classification
    """
    print("\n🧠 PLANNER NODE")
    print(f"Query: {state.query}")
    
    try:
        # Initialize LLM
        llm = ChatGroq(
            model_name=state.model_name,
            temperature=0,  # Deterministic for classification
            groq_api_key=settings.GROQ_API_KEY  # Uses GROQ_API_KEY from environment
        )
        
        # Create classification prompt
        prompt = ChatPromptTemplate.from_template("""
Classify this user query as either CONVERSATIONAL or TECHNICAL.

CONVERSATIONAL = Greetings, chitchat, general questions (Hi, How are you?, What's the weather?)
TECHNICAL = Asking about specific documents, data, or technical information

Query: {query}

Respond with ONLY the classification word: CONVERSATIONAL or TECHNICAL
""")
        
        # Classify
        chain = prompt | llm
        response = chain.invoke({"query": state.query})
        
        classification = response.content.strip().upper()
        
        # Ensure it's valid
        if "TECHNICAL" in classification:
            state.classification = "TECHNICAL"
            print(f"✅ Classification: TECHNICAL → Will retrieve documents")
        elif "CONVERSATIONAL" in classification:
            state.classification = "CONVERSATIONAL"
            print(f"✅ Classification: CONVERSATIONAL → Will respond directly")
        else:
            # Default to TECHNICAL if unclear
            state.classification = "TECHNICAL"
            print(f"⚠️ Unclear classification, defaulting to TECHNICAL")
        
        state.reasoning = f"Classified as {state.classification}"
        
    except Exception as e:
        print(f"❌ Error in planner: {e}")
        state.classification = "TECHNICAL"  # Default
        state.reasoning = f"Error during classification: {str(e)}"
    
    return state


if __name__ == "__main__":
    # Test the planner
    from app.config import settings
    
    # Test conversational
    state1 = AgentState(query="Hello! How are you?")
    result1 = planner_node(state1)
    print(f"Result: {result1.classification}\n")
    
    # Test technical
    state2 = AgentState(query="What does the document say about machine learning?")
    result2 = planner_node(state2)
    print(f"Result: {result2.classification}")