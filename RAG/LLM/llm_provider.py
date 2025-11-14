# RAG/LLM/llm_provider.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Protocol, Optional

class LLMClient(Protocol):
    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]: ...


class LocalHTTPClient:

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        auth_header: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "http://143.239.81.217:6010/generate")
        self.timeout_s = float(timeout_s or os.getenv("LOCAL_LLM_TIMEOUT_S", 60))
        self.auth_header = auth_header or os.getenv("LOCAL_LLM_AUTH")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        import requests
        context = "\n\n".join(
            f"Document {i+1}:\n{getattr(d, 'page_content', str(d))}"
            for i, d in enumerate(docs)
        )

        # question = "Who is the president of USA?"
        # context = "President is Kamala"

        print(f"question: {question}")
        print(f"context: {context}")

        payload = {"prompt": question, "context": context}
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header

        try:
            r = requests.post(self.base_url, json=payload, headers=headers, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            # Keep the error readable for upstream logs
            raise RuntimeError(f"LocalHTTPClient call failed: {e}") from e

        answer = data.get("answer")
        if not isinstance(answer, str):
            raise RuntimeError(f"LocalHTTPClient: unexpected response payload: {data}")
        return {"answer": answer, "sources": docs}

class DeepSeekClient:
    def __init__(self, model_slug: str | None = None, api_key: str | None = None, base_url: str | None = None):
        from openai import OpenAI  # lazy import
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("DeepSeek/OpenRouter requires OPENROUTER_API_KEY (or pass api_key to DeepSeekClient).")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model_slug or os.getenv("OR_MODEL_SLUG", "deepseek/deepseek-chat-v3-0324:free")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n\n".join(f"Document {i+1}:\n{getattr(d,'page_content',str(d))}" for i,d in enumerate(docs))
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=700,
            messages=[
                {"role":"system","content":"You are a precise, concise RAG assistant. Use ONLY the provided context. If unknown, say you don't know."},
                {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
        )
        return {"answer": completion.choices[0].message.content, "sources": docs}


class ClaudeClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # lazy import
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Claude/Anthropic requires ANTHROPIC_API_KEY (or pass api_key to ClaudeClient).")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n\n".join(f"Document {i+1}: {getattr(d,'page_content',str(d))}" for i,d in enumerate(docs))
        prompt = f"""You are a precise, concise RAG assistant. Use ONLY the provided context. If unknown, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            temperature=0.1,
            messages=[{"role":"user","content":prompt}],
        )
        return {"answer": msg.content[0].text, "sources": docs}

def build_llm(provider: str | None = None, **overrides) -> LLMClient:
    """
    Dependency-injection friendly factory.
    provider: "claude" | "deepseek" (case-insensitive). If None, checks env LLM_PROVIDER, else defaults to "claude".
    overrides: model=..., api_key=..., base_url=... (DeepSeek only)
    """
    prov = (provider or os.getenv("LLM_PROVIDER") or "claude").lower()
    if prov == "claude":
        return ClaudeClient(model=overrides.get("model"), api_key=overrides.get("api_key"))
    if prov == "deepseek":
        return DeepSeekClient(model_slug=overrides.get("model"),
                              api_key=overrides.get("api_key"),
                              base_url=overrides.get("base_url"))
    if prov == "local":
        return LocalHTTPClient(
            base_url=overrides.get("base_url"),
            timeout_s=overrides.get("timeout_s"),
            auth_header=overrides.get("auth_header"),
        )

    raise ValueError(f"Unknown provider: {provider}. Use 'claude', 'deepseek', or 'local'.")


def build_llm_from_env() -> LLMClient:
    """
    Backward-compatible env-based factory.
    """
    provider = (os.getenv("LLM_PROVIDER") or "claude").lower()
    return build_llm(provider)