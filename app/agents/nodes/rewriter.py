"""Rewriter Node - Turn follow-up questions into standalone ones using the chat history"""
from typing import List

from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from app.agents.llm import make_llm
from app.agents.state import AgentState


REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Rewrite the user's latest message so it can be understood without the conversation below.
Replace words like "it", "that", "they" or "the second one" with what they refer to. Otherwise keep the user's wording,
in their own voice: "I", "me", "my" and "you" stay as they are.
If the message already stands on its own, is small talk, or asks about this conversation itself
(What did I ask earlier?, What's my name?), return it unchanged.

CONVERSATION:
{history}

LATEST MESSAGE: {query}

Respond with ONLY the rewritten message.
""")


def format_history(messages: List[AnyMessage]) -> str:
    """Chat turns as 'User: ...' / 'Assistant: ...' lines"""
    return "\n".join(
        f"{'User' if message.type == 'human' else 'Assistant'}: {message.content}"
        for message in messages
    )


def rewriter_node(state: AgentState) -> AgentState:
    """
    Rewriter Node: Resolve a follow-up against the chat history.

    "Who are its authors?" after a question about a paper becomes
    "Who are the authors of <that paper>?", which retrieval can match.
    The first query of a chat is used as asked.

    Args:
        state: Current agent state (messages holds the chat history)

    Returns:
        Updated state with search_query
    """
    print("\n✏️ REWRITER NODE")
    state.search_query = state.query

    if not state.messages:
        print("No chat history → Using the query as asked")
        return state

    try:
        # Deterministic: the rewrite should only fill in references, not rephrase
        chain = REWRITE_PROMPT | make_llm(state.model_name, temperature=0)
        rewritten = chain.invoke({
            "history": format_history(state.messages),
            "query": state.query
        }).content.strip().strip('"')

        if rewritten and rewritten != state.query:
            state.search_query = rewritten
            state.reasoning = f'Read follow-up as "{rewritten}"'
            print(f"✅ Rewrote as: {rewritten}")
        else:
            print("✅ Query already stands on its own")

    except Exception as e:
        # Fall back to the query as asked
        print(f"❌ Error in rewriter: {e}")

    return state
