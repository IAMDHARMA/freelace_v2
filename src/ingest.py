# src/ingest.py

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

# Validate env vars at startup
DATABASE_URL = os.getenv("DATABASE_URL")
assert DATABASE_URL, "DATABASE_URL is not set in .env"

# ✅ FIXED: Same embedding model used in both ingest.py and rag.py
# Must match exactly — different models produce incompatible vector dimensions
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = PGVector(
    collection_name="ai_tutor_docs",
    connection=DATABASE_URL,
    embeddings=embeddings,
)

# Load original document
loader = PyPDFLoader("data/my_notes.pdf")
docs = loader.load()

# Split document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
split_docs = text_splitter.split_documents(docs)

# Store in PGVector
vectorstore.add_documents(split_docs)

print(f"✅ Ingested {len(split_docs)} chunks into PGVector successfully!")
