"""MongoDB Atlas helpers for the RAG learning project."""
import os
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
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


def list_rag_documents(collection: Collection, limit: int = 50) -> list[dict[str, Any]]:
    """Return recent RAG knowledge documents for display in the UI."""
    cursor = (
        collection.find(
            {"type": "rag_document"},
            {"text": 1, "entity": 1, "slot": 1, "knowledge_type": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    results = []
    for doc in cursor:
        results.append(
            {
                "id": str(doc.get("_id")),
                "text": doc.get("text", ""),
                "entity": doc.get("entity"),
                "slot": doc.get("slot"),
                "knowledge_type": doc.get("knowledge_type"),
                "created_at": doc.get("created_at"),
            }
        )
    return results


def delete_rag_document(collection: Collection, document_id: str) -> bool:
    """Delete a single RAG knowledge document by its id."""
    try:
        object_id = ObjectId(document_id)
    except Exception:
        return False

    result = collection.delete_one({"_id": object_id, "type": "rag_document"})
    return result.deleted_count > 0

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
    route: str | None = None,
) -> None:
    """Store a chat log with the RAG context used for the answer."""
    log_entry = {
        "type": "chat_log",
        "question": question,
        "answer": answer,
        "retrieved_documents": retrieved_documents,
        "created_at": datetime.now(timezone.utc),
    }
    if route:
        log_entry["route"] = route
    collection.insert_one(log_entry)
