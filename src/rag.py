# src/rag.py

import os
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

from .llm import llm  # ✅ FIXED: Import shared LLM — no duplicate, no model mismatch

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL is not set in .env"

COLLECTION_NAME = "ai_tutor_docs"


def get_session_history(session_id: str):
    """Return per-session PostgreSQL chat history store."""
    return PostgresChatMessageHistory(
        connection_string=DATABASE_URL,
        session_id=session_id,
        table_name="chat_history"
    )


def format_docs(docs: list) -> str:
    """
    ✅ FIXED: Stringify retrieved Document objects into plain text.
    Without this, raw Document objects are passed to the prompt template
    and rendered as Python repr strings, not their actual content.
    """
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)


def get_qa_chain():
    # ✅ FIXED: Same embedding model as ingest.py — must match or retrieval breaks
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embeddings=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a friendly English Tutor.\n"
            "The student may ask questions in their native language.\n"
            "Your job is to TEACH ENGLISH concepts: words, grammar, usage, and examples.\n\n"
            "CRITICAL RULES:\n"
            "- Always explain in {language} language so the student understands.\n"
            "- Do NOT switch fully to English in your explanation.\n"
            "- Use simple, beginner-friendly language.\n"
            "- Give complete, detailed answers with examples.\n"
            "- Never stop mid-sentence. Always finish your explanation fully.\n"
        ),
        (
            "system",
            "Relevant study material (use if helpful, ignore if empty or irrelevant):\n"
            "{context}"
        ),
        # ✅ FIXED: History is injected here by RunnableWithMessageHistory
        # Do NOT also set it in the inner lambda — that prevents injection from working
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ])

    # ✅ FIXED: Context is properly formatted from Document objects to plain text
    # ✅ FIXED: History key is NOT set in the inner dict — let RunnableWithMessageHistory handle it
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
            "language": RunnableLambda(lambda x: x.get("language", "English")),
        }
        | prompt
        | llm
    )

    # ✅ Wrap with conversation history — history_messages_key must match placeholder in prompt
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return conversational_chain
