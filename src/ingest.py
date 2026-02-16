from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = PGVector(
    collection_name="ai_tutor_docs",
    connection=DATABASE_URL,
    embeddings=embeddings,
)

docs = [
    Document(page_content="Dharmarajan completed BSc in Computer Science at Adam College of Arts and Science, Trichy."),
    Document(page_content="Artificial Intelligence focuses on machine learning, deep learning and neural networks."),
]

vectorstore.add_documents(docs)

print("Documents added successfully!")
