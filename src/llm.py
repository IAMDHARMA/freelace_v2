# src/llm.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Validate env vars at startup
assert os.getenv("GROQ_API_KEY"), "GROQ_API_KEY is not set in .env"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=1024,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
