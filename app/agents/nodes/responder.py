"""Responder Node - Generate the final answer"""
import re
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from app.config import settings
from app.agents.llm import make_llm
from app.agents.state import AgentState
from app.services.ranking_service import get_ranking_service


# Reply to a document question when the library has nothing that answers it
NOT_FOUND_RESPONSE = (
    "I couldn't find that in your documents. "
    "Try rephrasing the question, or upload a document that covers it."
)

# What the model replies when the passages don't contain the answer
NOT_FOUND_SENTINEL = "NOT_FOUND"

GROUNDED_PROMPT = ChatPromptTemplate.from_template("""
You answer questions using only the numbered passages below, taken from the user's documents.

PASSAGES:
{context}

QUESTION: {query}

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- Cite the passage behind every claim with its number in plain square brackets, like [1] or [2][3].
- If the passages only partly answer the question, answer that part and say what is missing.
- If the passages don't contain the answer at all, reply with exactly: NOT_FOUND

ANSWER:
""")

# Small talk sees the earlier turns, so "What did I ask you earlier?" works
CONVERSATIONAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful AI assistant."),
    MessagesPlaceholder("history"),
    ("human", "{query}")
])


def format_passages(docs: List[Dict]) -> str:
    """Number passages [1], [2], ... in the order the UI shows them as sources"""
    parts = []
    for number, doc in enumerate(docs, 1):
        location = doc["source"]
        if doc.get("page") is not None and doc["source"].lower().endswith(".pdf"):
            location += f", page {doc['page'] + 1}"
        parts.append(f"[{number}] Source: {location}\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


def normalize_citations(answer: str) -> str:
    """Rewrite citation styles like 【1】 or 【1†source】 as [1], matching the UI's source labels"""
    return re.sub(
        r"【\s*(\d+(?:\s*,\s*\d+)*)[^】]*】",
        lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(1))),
        answer
    )


def is_not_found(answer: str) -> bool:
    """True when the model signalled that the passages don't contain the answer"""
    return answer.strip().strip("*.`'\" ").upper() == NOT_FOUND_SENTINEL


def responder_node(state: AgentState) -> AgentState:
    """
    Responder Node: Generate the final answer based on:
    - Retrieved documents (if technical)
    - Classification (conversational vs technical)

    Steps:
    1. Rerank retrieved documents (if any)
    2. Number the top passages as context
    3. Generate a cited answer from those passages only (or say it wasn't found)
    4. Add the question and answer to the chat's memory

    Args:
        state: Current agent state

    Returns:
        Updated state with response
    """
    print("\n💬 RESPONDER NODE")
    print(f"Classification: {state.classification}")

    try:
        if state.classification == "CONVERSATIONAL":
            # Conversational - no documents needed
            chain = CONVERSATIONAL_PROMPT | make_llm(state.model_name, temperature=0.7)
            state.response = chain.invoke({"history": state.messages, "query": state.query}).content
            state.reasoning += " → Generated response"

        elif not state.retrieved_docs:
            # Document question, but nothing in the library cleared the relevance floor
            state.response = NOT_FOUND_RESPONSE
            state.reasoning += " → No relevant passages → Replied not found"

        else:
            # Document question - rerank, then answer only from the top passages
            print("Reranking documents...")
            doc_texts = [doc["text"] for doc in state.retrieved_docs]
            reranked = get_ranking_service().rerank(state.search_query, doc_texts, top_k=3)

            # If reranking fails, fall back to the closest passages by similarity
            if not reranked:
                reranked = [
                    {"text": doc["text"], "score": doc["score"], "index": i}
                    for i, doc in enumerate(state.retrieved_docs[:3])
                ]

            state.reranked_docs = [
                {
                    "text": r["text"],
                    "score": r["score"],
                    "source": state.retrieved_docs[r["index"]].get("source", "Unknown"),
                    "page": state.retrieved_docs[r["index"]].get("page")
                }
                for r in reranked
            ]
            print(f"✅ Reranked to {len(state.reranked_docs)} documents")

            # Temperature 0: stick to the passages rather than paraphrase creatively.
            # The standalone query carries what a follow-up refers to; earlier answers
            # stay out of this prompt so their [n] citations can't mix with these passages.
            chain = GROUNDED_PROMPT | make_llm(state.model_name, temperature=0)
            answer = chain.invoke({
                "query": state.search_query,
                "context": format_passages(state.reranked_docs)
            }).content

            if is_not_found(answer):
                state.response = NOT_FOUND_RESPONSE
                state.reasoning += f" → Reranked {len(state.reranked_docs)} docs → Passages didn't answer it → Replied not found"
                state.reranked_docs = []
            else:
                state.response = normalize_citations(answer)
                state.reasoning += f" → Reranked {len(state.reranked_docs)} docs → Generated cited response"

        print(f"✅ Generated response")
        print(f"   Length: {len(state.response)} characters")

    except Exception as e:
        print(f"❌ Error in responder: {e}")
        state.response = f"I encountered an error: {str(e)}"
        state.reasoning += f" → Error: {str(e)}"

    # Remember this turn; turns older than the last MEMORY_TURNS are forgotten
    turns = state.messages + [HumanMessage(state.query), AIMessage(state.response)]
    state.messages = [
        RemoveMessage(id=REMOVE_ALL_MESSAGES),
        *turns[max(0, len(turns) - 2 * settings.MEMORY_TURNS):]
    ]

    return state
