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
from typing import Any

# Web search / scraping libs
from googlesearch import search
from bs4 import BeautifulSoup
import requests

from db import (
    build_rag_context,
    delete_rag_document,
    get_collection,
    list_rag_documents,
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
    entity: str | None = Field(None, description="Target entity for the knowledge")
    slot: str | None = Field(None, description="Information type / slot name")
    type: str | None = Field(None, description="Knowledge type: fact / history / summary")


class ChatQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class ChatQueryResponse(BaseModel):
    answer: str
    retrieved_documents: list[dict[str, Any]]


class ChatRouteRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")

class ChatRouteResponse(BaseModel):
    answer: str
    retrieved_documents: list[dict[str, Any]]
    route: str


class RagDocumentResponse(BaseModel):
    id: str
    text: str
    entity: str | None
    slot: str | None
    type: str | None
    created_at: str | None


class RagListResponse(BaseModel):
    documents: list[RagDocumentResponse]


class BusinessRequest(BaseModel):
    supplierName: str
    productName: str


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
        "type": "rag_document",
        "text": payload.text,
        "entity": payload.entity,
        "slot": payload.slot,
        "knowledge_type": payload.type,
        "embedding": vector,
        "created_at": datetime.now(timezone.utc),
    }

    collection = get_collection()
    store_rag_document(collection, document)

    return {"message": "Knowledge stored successfully."}


@app.get("/rag/list", response_model=RagListResponse)
def list_rag_knowledge() -> RagListResponse:
    """Return recent RAG knowledge documents for the frontend list."""
    collection = get_collection()
    documents = list_rag_documents(collection, limit=50)
    formatted = [
        RagDocumentResponse(
            id=doc["id"],
            text=doc["text"],
            entity=doc.get("entity"),
            slot=doc.get("slot"),
            type=doc.get("knowledge_type"),
            created_at=doc["created_at"].isoformat() if doc["created_at"] else None,
        )
        for doc in documents
    ]
    return RagListResponse(documents=formatted)


@app.delete("/rag/{document_id}")
def delete_rag_knowledge(document_id: str) -> dict[str, str]:
    """Delete a single RAG knowledge document by id."""
    collection = get_collection()
    deleted = delete_rag_document(collection, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted."}


@app.post("/chat/route", response_model=ChatQueryResponse)
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


@app.post("/chat/query", response_model=ChatRouteResponse)
def chat_route(payload: ChatRouteRequest) -> ChatRouteResponse:
    """
    Simple query routing endpoint.
    If the top retrieved score is below the threshold, fall back to a plain LLM answer.
    Otherwise use RAG context.
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

    THRESHOLD = 0.67
    top_score = retrieved[0]["score"] if retrieved else 0.0
    use_rag = top_score >= THRESHOLD
    route = "rag" if use_rag else "llm"

    system_prompt = (
        "You are a helpful assistant. Use the provided context when relevant, "
        "and say when the context does not contain the answer."
    )

    if use_rag:
        context_text = "\n".join([f"- {doc['text']}" for doc in retrieved])
        user_prompt = (
            "Context:\n"
            f"{context_text}\n\n"
            "Question:\n"
            f"{payload.question}\n\n"
            "Answer in Korean to match the learning UI."
        )
    else:
        user_prompt = (
            "Question:\n"
            f"{payload.question}\n\n"
            "Answer in Korean to match the learning UI."
        )
    if route == "rag":
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
    elif route == "llm": 
        answer = "사용자님이 주신 요청에 부합한 지식이 없는 것 같습니다. 죄송합니당"

    log_chat(
        collection=collection,
        question=payload.question,
        answer=answer,
        retrieved_documents=retrieved if use_rag else [],
        route=route,
    )

    return ChatRouteResponse(
        answer=answer,
        retrieved_documents=retrieved if use_rag else [],
        route=route,
    )

from typing import Any
import re

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS  # pip install duckduckgo-search

class BusinessRequest(BaseModel):
    supplierName: str
    productName: str

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "close",
}

def ddg_search(query: str, k: int = 5) -> list[str]:
    urls: list[str] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            href = r.get("href")
            if href:
                urls.append(href)
    return urls

def extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text

def fetch_page_text(session: requests.Session, url: str, timeout: int = 10) -> str | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()

        ctype = (resp.headers.get("content-type") or "").lower()
        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            return None

        resp.encoding = resp.apparent_encoding or resp.encoding
        return extract_visible_text(resp.text)
    except Exception:
        return None

from db import get_collection, store_rag_documents 
# ---- 텍스트 chunk 유틸 (임베딩 길이 제한/비용 방지용) ----
def chunk_text(text: str, max_chars: int = 6000, overlap: int = 200) -> list[str]:
    """
    아주 단순한 char 기반 청킹.
    - max_chars: 한 덩어리 최대 길이
    - overlap: 문맥 이어붙이기용 겹침
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)

        # 문단 경계에서 끊기 (가능하면)
        window = text[start:end]
        cut = window.rfind("\n\n")
        if cut > max_chars * 0.6:
            end = start + cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks

@app.post("/business/rag/ingest")
def business_rag_ingest(payload: BusinessRequest) -> dict[str, Any]:
    """
    1) supplierName+productName로 검색/크롤링
    2) 문서 chunk
    3) chunk들을 embedding
    4) MongoDB Atlas에 저장 (embedding 필드 포함)
    """
    query = f"{payload.supplierName} {payload.productName}".strip()

    # 1) 검색/크롤링은 기존 business_rag 로직 그대로 재사용하는 게 깔끔하지만,
    #    여기서는 핵심만: ddg_search + fetch_page_text 사용(현재 파일에 이미 있음) :contentReference[oaicite:4]{index=4}
    urls = ddg_search(query, k=5)
    if not urls:
        raise HTTPException(status_code=502, detail="Search returned 0 results (likely blocked or network issue).")

    fetched_docs: list[dict[str, str]] = []
    with requests.Session() as session:
        for url in urls:
            page_text = fetch_page_text(session, url)
            if not page_text:
                continue
            fetched_docs.append({"url": url, "text": page_text})

    if not fetched_docs:
        raise HTTPException(status_code=502, detail="All fetch attempts failed (blocked/timeout/non-html).")

    # 2) chunk
    items: list[dict[str, Any]] = []
    for d in fetched_docs:
        for i, ch in enumerate(chunk_text(d["text"], max_chars=6000, overlap=200)):
            items.append({
                "url": d["url"],
                "chunk_index": i,
                "text": ch,
            })

    if not items:
        raise HTTPException(status_code=502, detail="Fetched text was empty after processing.")

    # 3) embeddings (배치로)
    client = OpenAI()
    try:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=[it["text"] for it in items],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    vectors = [row.embedding for row in emb.data]

    # 4) MongoDB 저장 (Atlas Vector Search는 'embedding' path를 쓰는 걸로 이미 설정돼있음) :contentReference[oaicite:5]{index=5}
    now = datetime.now(timezone.utc)
    docs_to_insert: list[dict[str, Any]] = []
    for it, vec in zip(items, vectors):
        docs_to_insert.append({
            "type": "rag_document",
            "text": it["text"],
            "embedding": vec,

            # 메타데이터(원하면 더 늘려도 됨)
            "entity": payload.supplierName,
            "slot": payload.productName,
            "knowledge_type": "web",

            "source_url": it["url"],
            "chunk_index": it["chunk_index"],
            "query": query,

            "created_at": now,
        })

    collection = get_collection()
    inserted_count = store_rag_documents(collection, docs_to_insert)

    return {
        "query": query,
        "urls": urls,
        "fetched_pages": len(fetched_docs),
        "chunks": len(items),
        "inserted": inserted_count,
    }