"""
FastAPI backend for a learning project that compares RAG vs plain LLM responses.
We keep the code intentionally simple and heavily commented for study.
"""
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from .db import (
    build_rag_context,
    get_collection,
    log_chat,
    store_rag_document,
)

app = FastAPI(title="RAG vs LLM Learning API")

# Allow local frontend to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    ,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class RagStoreRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Knowledge text to store for RAG")


class ChatQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class ChatQueryResponse(BaseModel):
    answer: str
    retrieved_documents: list[dict[str, Any]]


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rag/store")
def store_rag_knowledge(payload: RagStoreRequest) -> dict[str, str]:
    """
    Store user-provided knowledge for RAG.
    Steps:
      1) Embed the text with OpenAI embeddings.
      2) Save the text + embedding vector to MongoDB Atlas.
    """
    client = OpenAI()
    try:
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=payload.text,
        )
    except Exception as exc:  # OpenAI can raise multiple exception types
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    vector = embedding_response.data[0].embedding
    document = {
        "text": payload.text,
        "embedding": vector,
        "created_at": datetime.now(timezone.utc),
    }

    collection = get_collection()
    store_rag_document(collection, document)

    return {"message": "Knowledge stored successfully."}


@app.post("/chat/query", response_model=ChatQueryResponse)
def chat_query(payload: ChatQueryRequest) -> ChatQueryResponse:
    """
    RAG-enabled Q&A endpoint.
    Steps:
      1) Embed the question.
      2) Retrieve similar documents from MongoDB Atlas Vector Search.
      3) Build a context block.
      4) Ask the LLM to answer with the context.
      5) Save the chat log for later study.
    """
    client = OpenAI()

    try:
        embedding_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=payload.question,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    question_vector = embedding_response.data[0].embedding

    collection = get_collection()
    retrieved = build_rag_context(collection, question_vector, limit=3)
    context_text = "\n".join([f"- {doc['text']}" for doc in retrieved])

    system_prompt = (
        "You are a helpful assistant. Use the provided context when relevant, "
        "and say when the context does not contain the answer."
    )

    user_prompt = (
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{payload.question}\n\n"
        "Answer in Korean to match the learning UI."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat completion failed: {exc}") from exc

    answer = completion.choices[0].message.content

    log_chat(
        collection=collection,
        question=payload.question,
        answer=answer,
        retrieved_documents=retrieved,
    )

    return ChatQueryResponse(answer=answer, retrieved_documents=retrieved)
