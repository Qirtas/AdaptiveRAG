# RAG/LLM/deepseek_integration.py
from openai import OpenAI
import os, time

MODEL_SLUG = os.getenv("OR_MODEL_SLUG", "deepseek/deepseek-chat-v3-0324:free")

_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def ask_deepseek_with_docs(question, docs, max_tokens=700, temperature=0.2):
    """
    Function to query DeepSeek (via OpenRouter) with pre-filtered documents.
    Mirrors the signature & return shape of ask_claude_with_docs.
    """
    # Step 1: Format context from provided documents
    context = "\n\n".join([
        f"Document {i + 1}:\n{getattr(doc, 'page_content', str(doc))}"
        for i, doc in enumerate(docs)
    ])

    # Step 2: Query DeepSeek
    completion = _client.chat.completions.create(
        model=MODEL_SLUG,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise, concise RAG assistant. "
                    "Use ONLY the provided context. If the answer is not in the context, say you don't know. "
                    "Cite evidence inline as [Doc i] when helpful."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            },
        ],
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "sources": docs
    }