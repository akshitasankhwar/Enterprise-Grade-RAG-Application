"""Planner Node - Classify the query and decide routing"""
from langchain_core.prompts import ChatPromptTemplate
from app.agents.llm import make_llm
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
    print(f"Query: {state.search_query}")
    
    try:
        # Deterministic for classification
        llm = make_llm(state.model_name, temperature=0)

        # Create classification prompt
        prompt = ChatPromptTemplate.from_template("""
You route messages for an assistant that answers questions from the user's uploaded documents.
Classify this user query as either CONVERSATIONAL or TECHNICAL.

CONVERSATIONAL = Greetings, thanks, small talk, or questions about the assistant or this conversation itself (Hi, How are you?, Thanks!, What can you do?, What did I ask you earlier?)
TECHNICAL = Any request for information, facts, numbers, rules, people, dates or summaries, even if it doesn't mention documents (What is the meal allowance?, Who leads the project?, Summarize the report)

When unsure, answer TECHNICAL.

Query: {query}

Respond with ONLY the classification word: CONVERSATIONAL or TECHNICAL
""")
        
        # Classify
        chain = prompt | llm
        response = chain.invoke({"query": state.search_query})
        
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
        
        # The rewriter may have started the reasoning with how it read a follow-up
        state.reasoning += f"{' → ' if state.reasoning else ''}Classified as {state.classification}"
        
    except Exception as e:
        print(f"❌ Error in planner: {e}")
        state.classification = "TECHNICAL"  # Default
        state.reasoning += f"{' → ' if state.reasoning else ''}Error during classification: {str(e)}"
    
    return state
