"""MongoDB Atlas helpers for the RAG learning project."""
import os
from datetime import datetime, timezone
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv


load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "")
MONGODB_DB = os.getenv("MONGODB_DB", "rag_learning")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "rag_documents")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "rag_vector_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


_client: MongoClient | None = None


def get_collection() -> Collection:
    """
    Create (or reuse) a MongoDB client.
    The URI is read from environment variables to keep secrets out of code.
    """
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI is not set")

    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)

    return _client[MONGODB_DB][MONGODB_COLLECTION]


def store_rag_document(collection: Collection, document: dict[str, Any]) -> None:
    """Insert a new RAG knowledge document into MongoDB."""
    collection.insert_one(document)


def build_rag_context(
    collection: Collection,
    query_vector: list[float],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """
    Use MongoDB Atlas Vector Search to retrieve similar documents.
    The vector index is assumed to exist already.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": 50,
                "limit": limit,
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def log_chat(
    collection: Collection,
    question: str,
    answer: str,
    retrieved_documents: list[dict[str, Any]],
) -> None:
    """Store a chat log with the RAG context used for the answer."""
    log_entry = {
        "type": "chat_log",
        "question": question,
        "answer": answer,
        "retrieved_documents": retrieved_documents,
        "created_at": datetime.now(timezone.utc),
    }
    collection.insert_one(log_entry)
