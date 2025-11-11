import os

import anthropic


def ask_claude_with_docs(question, docs):
    """
    Function to query Claude with pre-filtered documents
    """
    # Step 1: Format context from provided documents
    context = "\n\n".join([
        f"Document {i + 1}: {doc.page_content}"
        for i, doc in enumerate(docs)
    ])

    # Step 2: Query Claude
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # client = anthropic.Anthropic(api_key="")

    prompt = f"""Answer this question based on the provided context:

Context:
{context}

Question: {question}

Answer:"""

    print(f"Prompt for LLM: {prompt}")
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": message.content[0].text,
        "sources": docs
    }