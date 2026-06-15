# retrieval/retriever.py
import os
import re
import json
import logging
from collections import defaultdict
from typing import AsyncGenerator

from qdrant_client import models  
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

log = logging.getLogger(__name__)

session_store: dict[str, list] = defaultdict(list)
session_sources: dict[str, list] = defaultdict(list)

MAX_HISTORY = 10


# ══════════════════════════════════════════════════════════════════════
# VECTORSTORE
# ══════════════════════════════════════════════════════════════════════
def get_vectorstore(embeddings):
    try:
        vs = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=os.getenv("QDRANT_COLLECTION"),
        )
        log.info("Connected to Qdrant.")
        return vs
    except Exception as e:
        log.error(f"Qdrant connection failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════
def format_history(messages: list) -> str:
    if not messages:
        return "No previous conversation."
    lines = []
    for msg in messages[-MAX_HISTORY:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def save_to_history(session_id: str, question: str, answer: str):
    session_store[session_id].append(HumanMessage(content=question))
    session_store[session_id].append(AIMessage(content=answer))
    if len(session_store[session_id]) > MAX_HISTORY * 2:
        session_store[session_id] = session_store[session_id][-MAX_HISTORY * 2:]


def get_session_sources(session_id: str) -> list:
    return session_sources.get(session_id, [])


def update_session_sources(session_id: str, docs: list):
    existing = {s["source"] for s in session_sources[session_id]}
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        source_type = doc.metadata.get("source_type", "unknown")
        if source not in existing:
            session_sources[session_id].append({
                "source": source,
                "source_type": source_type,
            })
            existing.add(source)


# ══════════════════════════════════════════════════════════════════════
# CLEAR COLLECTION — new endpoint support
# WHY: Demo/testing needs clean slate without deleting Qdrant collection
# Deletes all points for a specific session only
# ══════════════════════════════════════════════════════════════════════
def clear_session_data(session_id: str) -> dict:
    """
    WHY WIPE EVERYTHING:
    Single user app — clear means fresh start
    Delete ALL vectors from collection + all session memory
    User starts completely clean
    """
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        collection = os.getenv("QDRANT_COLLECTION")

        # Get point count before delete
        info = client.get_collection(collection)
        total_before = info.points_count

        # Delete ALL points in collection
        # WHY delete_vectors not recreate_collection:
        # recreate_collection resets indexes too — need to recreate them
        # Deleting all points keeps collection + indexes intact
        # Faster and safer on free tier
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[]  # empty filter = match ALL points
                )
            ),
        )

        # Clear all in-memory session data
        session_store.clear()
        session_sources.clear()

        log.info(f"Collection cleared: {total_before} points deleted")
        return {
            "status": "cleared",
            "points_deleted": total_before,
        }

    except Exception as e:
        log.error(f"Clear collection failed: {e}")
        raise

# ══════════════════════════════════════════════════════════════════════
# PROMPT
# ══════════════════════════════════════════════════════════════════════
def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(
        "You are a helpful AI learning assistant for Samasocial.\n"
        "Answer questions based ONLY on the provided document context.\n\n"
        "Previous conversation:\n{history}\n\n"
        "Context from documents:\n{context}\n\n"
        "Rules:\n"
        "- Answer ONLY from the context above\n"
        "- ALWAYS cite your source e.g. [Page 3 of notes.pdf] or [at 2:30 in the video]\n"
        "- If the answer is not in the context, say: "
        "\"I don't have that information in the provided sources.\"\n"
        "- Explain concepts simply when asked\n"
        "- Decline out-of-scope questions politely\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )


# ══════════════════════════════════════════════════════════════════════
# RERANKER
# WHY CROSS ENCODER NOT BI ENCODER:
# Bi-encoder (embedding) = fast but approximate
# Cross-encoder = slower but scores query+doc together = more accurate
# We retrieve 10 candidates, rerank to top 5
# WHY ms-marco: lightweight, runs on CPU, no GPU needed on free tier
# ══════════════════════════════════════════════════════════════════════
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            model = HuggingFaceCrossEncoder(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            _reranker = CrossEncoderReranker(model=model, top_n=5)
            log.info("Reranker initialized.")
        except Exception as e:
            log.warning(f"Reranker init failed, skipping: {e}")
            _reranker = False
    return _reranker if _reranker else None


# ══════════════════════════════════════════════════════════════════════
# RETRIEVE + RERANK
# ══════════════════════════════════════════════════════════════════════
def retrieve_and_build_context(vs, question: str, session_id: str):
    session_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=session_id),
            )
        ]
    )

    # Fetch more candidates for reranker to work with
    base_retriever = vs.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": session_filter,
        }
    )

    # Apply reranker if available
    reranker = get_reranker()
    if reranker:
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
    else:
        retriever = base_retriever

    try:
        docs = retriever.invoke(question)
    except Exception as e:
        log.error(f"Retrieval failed: {e}")
        raise

    if not docs:
        return "No relevant content found in your uploaded documents.", [], []

    context_parts = []
    citations = []

    for doc in docs:
        citation = doc.metadata.get(
            "citation",
            doc.metadata.get("source_file", "Unknown source")
        )
        citations.append(citation)
        context_parts.append(f"[{citation}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    update_session_sources(session_id, docs)

    sources = list(set(
        doc.metadata.get("source_file", "Unknown")
        for doc in docs
    ))

    return context, citations, sources


# ══════════════════════════════════════════════════════════════════════
# QUERY — Non-streaming
# ══════════════════════════════════════════════════════════════════════
def query(
    question: str,
    embeddings,
    llm,
    session_id: str = "default",
) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vs = get_vectorstore(embeddings)
    if vs is None:
        raise ConnectionError("Could not connect to Qdrant.")

    history_text = format_history(session_store[session_id])
    context, citations, sources = retrieve_and_build_context(
        vs, question, session_id
    )

    prompt = build_prompt()
    chain = prompt | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "history": history_text,
            "context": context,
            "question": question,
        })
    except Exception as e:
        log.error(f"LLM chain failed: {e}")
        raise

    save_to_history(session_id, question, answer)
    log.info(f"Query answered — session: {session_id}")

    return {
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "session_id": session_id,
    }


# ══════════════════════════════════════════════════════════════════════
# STREAM QUERY
# FIX: Streaming was stripping spaces between tokens
# WHY: Azure OpenAI returns tokens with leading spaces e.g. " the", " answer"
# Previously we were stripping them — now preserve as-is
# ══════════════════════════════════════════════════════════════════════
async def stream_query(
    question: str,
    embeddings,
    llm,
    session_id: str = "default",
) -> AsyncGenerator[str, None]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    vs = get_vectorstore(embeddings)
    if vs is None:
        raise ConnectionError("Could not connect to Qdrant.")

    history_text = format_history(session_store[session_id])
    context, citations, sources = retrieve_and_build_context(
        vs, question, session_id
    )

    prompt = build_prompt()
    formatted = prompt.format_messages(
        history=history_text,
        context=context,
        question=question,
    )

    full_answer = ""

    try:
        async for chunk in llm.astream(formatted):
            token = chunk.content
            # WHY NOT STRIP:
            # Azure OpenAI sends " word" with leading space
            # Stripping = words run together "theanswer"
            # Preserve token exactly as received
            if token is not None:
                full_answer += token
                yield token

    except Exception as e:
        log.error(f"Stream failed: {e}")
        yield "[ERROR] Streaming failed."
        return

    save_to_history(session_id, question, full_answer)

    # Send metadata as final event
    yield f"\n[SOURCES]{json.dumps({'sources': sources, 'citations': citations, 'session_id': session_id})}"


# ══════════════════════════════════════════════════════════════════════
# QUIZ — Fixed to use session content not generic retrieval
# FIX: Was retrieving from entire collection not session docs
# Now filters by session_id so quiz is about uploaded content only
# ══════════════════════════════════════════════════════════════════════
async def generate_quiz_questions(
    session_id: str,
    embeddings,
    llm,
    num_questions: int = 5,
) -> list[dict]:
    vs = get_vectorstore(embeddings)
    if vs is None:
        raise ConnectionError("Could not connect to Qdrant.")

    # WHY FILTER BY SESSION:
    # Single user but session groups uploaded docs
    # Quiz only about what user uploaded this session
    session_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.session_id",
                match=MatchValue(value=session_id),
            )
        ]
    )

    retriever = vs.as_retriever(
        search_kwargs={
            "k": 10,
            "filter": session_filter,
        }
    )

    try:
        docs = retriever.invoke("key concepts main topics definitions")
    except Exception as e:
        log.error(f"Quiz retrieval failed: {e}")
        raise

    if not docs:
        raise ValueError("No content found. Please upload a document first.")

    context = "\n\n".join(doc.page_content for doc in docs[:8])

    prompt = ChatPromptTemplate.from_template(
        "You are a quiz generator.\n"
        "Generate exactly {num_questions} multiple choice questions "
        "based ONLY on the content below.\n\n"
        "Content:\n{context}\n\n"
        "STRICT RULES:\n"
        "- Questions must test understanding of the CONTENT only\n"
        "- Do NOT ask about file names, document titles, or metadata\n"
        "- Do NOT ask 'what file was uploaded' or similar\n"
        "- Each question must have exactly 4 options\n"
        "- Return ONLY raw JSON array, no markdown backticks\n\n"
        "[\n"
        "  {{\n"
        "    \"question\": \"...\",\n"
        "    \"options\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"],\n"
        "    \"correct\": \"A\",\n"
        "    \"explanation\": \"...\",\n"
        "    \"difficulty\": \"easy|medium|hard\"\n"
        "  }}\n"
        "]\n\n"
        "JSON:"
    )

    chain = prompt | llm | StrOutputParser()

    try:
        result = chain.invoke({
            "context": context,
            "num_questions": num_questions,
        })

        # Strip markdown if LLM adds backticks
        result = re.sub(r'```json|```', '', result).strip()

        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if not json_match:
            raise ValueError("LLM did not return valid JSON array")

        questions = json.loads(json_match.group())
        log.info(f"Generated {len(questions)} quiz questions")
        return questions

    except json.JSONDecodeError as e:
        log.error(f"Quiz JSON parse failed: {e}")
        raise ValueError("Failed to parse quiz — try again.")
    except Exception as e:
        log.error(f"Quiz generation failed: {e}")
        raise