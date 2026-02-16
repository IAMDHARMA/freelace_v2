# src/rag.py

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = "ai_tutor_docs"

# 🔥 Shared LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=200,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# 🔥 Store chat sessions
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=DATABASE_URL,
        session_id=session_id,
        table_name="chat_history"
    )


def get_qa_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PGVector(
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embeddings=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly AI Tutor. "
     "Explain concepts clearly, step-by-step, and in a simple way. "
     "Encourage curiosity and learning."),
    
    ("system",
     "Conversation History:\n{history}"),
    
    ("system",
     "Retrieved Context:\n{context}"),
    
    ("system",
     "Instructions:\n"
     "- If the retrieved context contains relevant information, use it in your answer.\n"
     "- If the context is empty or not relevant, answer using your general knowledge.\n"
     "- Keep responses clear, concise, and beginner-friendly.\n"
     "- If unsure, say so honestly."),
    
    ("human", "{question}")
])


    rag_chain = (
        {
        "context": RunnableLambda(lambda x: x["question"]) | retriever,
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: x.get("history", ""))
        }
        | prompt
        | llm
    )

    # ✅ Add Conversation History
    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return conversational_chain
