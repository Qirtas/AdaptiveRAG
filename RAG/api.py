# RAG/api.py
import os
import re
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.docstore.document import Document
from pydantic import BaseModel

from RAG.LLM.rag_controller import rag_with_validation
from RAG.Retrieval.retriever import setup_retriever
from RAG.LLM.llm_provider import build_llm_from_env  # <-- NEW

app = FastAPI(title="Datamite RAG API", version="0.1.0")

class AskRequest(BaseModel):
    question: str
    k: int = 3
    min_similarity: float = 0.48

class SourceItem(BaseModel):
    name: str | None = None
    score: float | None = None
    url: str | None = None
    snippet: str | None = None
    metadata: Dict[str, Any] | None = None

class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = []
    meta: Dict[str, Any] = {}

def _to_source_item(s: Any) -> Dict[str, Any]:
    if isinstance(s, Document):
        name = s.metadata.get("name") or s.metadata.get("id") or s.metadata.get("type")
        return {
            "name": name,
            "url": s.metadata.get("url"),
            "snippet": (s.page_content or "")[:300],
            "metadata": s.metadata
        }
    return {
        "name": None,
        "url": None,
        "snippet": str(s)[:300],
        "metadata": {}
    }

# ------------ App state ------------
retriever = None

@app.on_event("startup")
def _startup():
    """
    Load the retriever and build the chosen LLM ONCE when the server boots.
    """
    global retriever
    persist_dir = os.getenv("PERSIST_DIR", "RAG/ProcessedDocuments/chroma_db")
    k_default = 3

    try:
        retriever = setup_retriever(persist_directory=persist_dir, k=k_default)
        print(f"[startup] Retriever ready (persist_dir={persist_dir})")
    except Exception as e:
        print(f"[startup] Failed to setup retriever from {persist_dir}: {e}")

    # Build & inject the LLM implementation (DeepSeek or Claude)
    try:
        app.state.llm = build_llm_from_env()
        print(f"[startup] LLM provider ready: {os.getenv('LLM_PROVIDER', 'deepseek')}")
    except Exception as e:
        print(f"[startup] LLM setup failed: {e}")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not hasattr(app.state, "llm"):
        raise HTTPException(status_code=500, detail="LLM provider is not configured.")

    try:
        result = rag_with_validation(
            req.question,
            min_similarity=req.min_similarity,
            persist_directory=os.getenv("PERSIST_DIR", "RAG/ProcessedDocuments/chroma_db"),
            k=req.k,
            retriever=retriever,
            llm=app.state.llm,  # <-- INJECTED
        )
        raw_sources = result.get("sources", []) or []
        sources = [_to_source_item(s) for s in raw_sources]
        return {
            "answer": result.get("answer", ""),
            "sources": sources,
            "meta": {"k": req.k, "min_similarity": req.min_similarity}
        }
    except Exception as e:
        safe = re.sub(r'[^\x00-\x7F]+', '?', str(e))
        raise HTTPException(status_code=500, detail=f"RAG failed: {safe}")
