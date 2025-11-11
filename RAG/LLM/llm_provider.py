# RAG/LLM/llm_provider.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Protocol

# -------- Interface --------
class LLMClient(Protocol):
    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]: ...

# -------- DeepSeek via OpenRouter --------
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
                {"role":"system","content":"You are a precise, concise RAG assistant. Use ONLY the provided context. If unknown, say you don't know. Cite as [Doc i] when helpful."},
                {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ],
        )
        return {"answer": completion.choices[0].message.content, "sources": docs}

# -------- Claude via Anthropic --------
class ClaudeClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import anthropic  # lazy import

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Claude/Anthropic requires ANTHROPIC_API_KEY (or pass api_key to ClaudeClient).")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250929")

    def ask_with_docs(self, question: str, docs: List[Any]) -> Dict[str, Any]:
        context = "\n\n".join(f"Document {i+1}: {getattr(d,'page_content',str(d))}" for i,d in enumerate(docs))
        prompt = f"""Answer this question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.1,
            messages=[{"role":"user","content":prompt}],
        )
        return {"answer": msg.content[0].text, "sources": docs}

# -------- Factories --------
def build_llm(provider: str | None = None, **overrides) -> LLMClient:

    prov = (provider or os.getenv("LLM_PROVIDER") or "claude").lower()
    if prov == "claude":
        return ClaudeClient(model=overrides.get("model"), api_key=overrides.get("api_key"))
    if prov == "deepseek":
        return DeepSeekClient(model_slug=overrides.get("model"),
                              api_key=overrides.get("api_key"),
                              base_url=overrides.get("base_url"))
    raise ValueError(f"Unknown provider: {provider}. Use 'claude' or 'deepseek'.")

def build_llm_from_env() -> LLMClient:
    """
    Backward-compatible env-based factory.
    """
    provider = (os.getenv("LLM_PROVIDER") or "claude").lower()
    return build_llm(provider)