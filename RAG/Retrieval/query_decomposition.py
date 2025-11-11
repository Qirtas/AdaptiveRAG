# RAG/decompose/query_decomposition.py

import json
import re
from typing import Dict, List, Tuple, Any
from RAG.LLM.llm_provider import build_llm, LLMClient
from RAG.LLM.llm_provider import build_llm, LLMClient
from RAG.Retrieval.adaptive_validate import adaptive_rag_with_validation

def detect_multi_question(text: str, max_chars: int = 6000, max_parts: int = 5) -> Tuple[bool, List[str]]:
    """
    Local detector to flag multi-question queries using heuristics.

    Args:
        text: Input query text
        max_chars: Maximum characters to analyze (trim if longer)
        max_parts: Maximum number of parts to split into

    Returns:
        Tuple of (needs_decompose, candidate_splits)
    """
    # Trim input if too long
    original_length = len(text)
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"[Decompose] Trimmed input from {original_length} to {max_chars} chars")

    # Clean text for analysis
    cleaned_text = text.strip()

    # Count question marks
    qmarks = cleaned_text.count('?')

    # Check for bullet/numbered lists
    bullet_patterns = [
        r'^\s*[-•]\s+',  # - or •
        r'^\s*\d+[\.)]\s+',  # 1. or 1)
    ]
    has_bullets = any(re.search(pattern, line) for pattern in bullet_patterns for line in cleaned_text.split('\n'))

    # Check for multiple clauses ending with '?' OR semicolon-separated questions
    # Split by line breaks and semicolons
    clauses = []
    for line in cleaned_text.split('\n'):
        clauses.extend([clause.strip() for clause in line.split(';')])

    question_clauses = [clause for clause in clauses if clause.strip().endswith('?')]

    # Also check for semicolon-separated questions (even if they don't all end with '?')
    semicolon_parts = [part.strip() for part in cleaned_text.split(';') if part.strip()]
    has_semicolon_questions = len(semicolon_parts) >= 2 and any('?' in cleaned_text for _ in [1])

    # Exception: single analytical questions (trade-offs, comparisons, etc.)
    analytical_keywords = ['trade-off', 'tradeoff', 'compare', 'comparison', 'versus', ' vs ', ' vs.']
    is_analytical = any(keyword.lower() in cleaned_text.lower() for keyword in analytical_keywords)

    # Decision logic
    needs_decompose = False
    if qmarks >= 2 and not (is_analytical and qmarks == 2):
        needs_decompose = True
    elif has_bullets:
        needs_decompose = True
    elif len(question_clauses) >= 2:
        needs_decompose = True
    elif has_semicolon_questions and len(semicolon_parts) >= 2:
        needs_decompose = True

    # Generate candidate splits (best effort)
    if needs_decompose:
        candidate_splits = _generate_candidate_splits(cleaned_text, max_parts)
    else:
        candidate_splits = [cleaned_text]

    print(f"[Decompose] needs={needs_decompose} qmarks={qmarks} parts={len(candidate_splits)}")

    return needs_decompose, candidate_splits


def _generate_candidate_splits(text: str, max_parts: int) -> List[str]:
    """
    Generate candidate question splits using simple heuristics.
    """
    splits = []

    # Try splitting by question marks first
    question_parts = [part.strip() + '?' for part in text.split('?') if part.strip()]
    if question_parts and question_parts[-1] == '?':
        question_parts = question_parts[:-1]  # Remove empty last part

    if len(question_parts) > 1:
        splits = question_parts
    else:
        # Try splitting by bullets/numbers
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.endswith('?') or any(marker in line for marker in ['-', '•', '1.', '2.', '3.'])):
                # Clean up bullet points
                clean_line = re.sub(r'^\s*[-•]\s*', '', line)
                clean_line = re.sub(r'^\s*\d+[\.)]\s*', '', clean_line)
                if clean_line:
                    splits.append(clean_line)

        if not splits:
            # Fallback: split by semicolons or line breaks
            splits = [part.strip() for part in text.replace(';', '\n').split('\n') if part.strip()]

    # Cap to max_parts, merge remainder
    if len(splits) > max_parts:
        remainder = splits[max_parts:]
        splits = splits[:max_parts - 1]
        additional = "Additionally: " + "; ".join(remainder)
        splits.append(additional)

    return splits if splits else [text]


def decompose_to_json(submitted_text: str, provider: str = "claude", model: str = None, api_key: str = None) -> List[
    str]:
    """
    Use LLM to decompose multi-question text into clean JSON sub-questions.

    Args:
        submitted_text: The user's input text
        provider: LLM provider ("claude", "deepseek", etc.)
        model: Specific model to use (optional)
        api_key: API key for the LLM

    Returns:
        List of clean sub-questions (max 5)
    """
    prompt = f"""Task: If the input contains multiple user questions, extract up to 5 concise sub-questions.
Rules:
- Return JSON: {{"sub_questions": ["...", "..."]}}
- Each item must be a single, standalone question (<= 200 chars).
- Merge near-duplicates; keep order as in input.
- If only one question, return it as a single-item list.
Input:
{submitted_text}"""

    try:
        llm = build_llm(provider=provider, model=model, api_key=api_key)

        # Try different LLM call methods based on the client type
        if hasattr(llm, 'invoke'):
            response = llm.invoke(prompt)
        elif hasattr(llm, 'generate'):
            response = llm.generate(prompt)
        elif hasattr(llm, '__call__'):
            response = llm(prompt)
        else:
            raise AttributeError(f"LLM client has no known call method")

        # Try to parse JSON
        response_text = response.strip()

        # Handle potential markdown code blocks
        if isinstance(response_text, str) and "```json" in response_text.lower():
            json_start = response_text.lower().find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end].strip()
        elif isinstance(response_text, str) and "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                response_text = response_text[json_start:json_end].strip()

        parsed = json.loads(response_text)
        sub_questions = parsed.get("sub_questions", [])

        # Validate and cap length
        clean_questions = []
        for q in sub_questions[:5]:  # Max 5 questions
            if isinstance(q, str) and q.strip():
                clean_q = q.strip()[:200]  # Max 200 chars
                clean_questions.append(clean_q)

        return clean_questions if clean_questions else [submitted_text]

    except Exception as e:
        print(f"[Decompose] LLM decomposition failed: {e}")
        # Fallback to detector splits
        _, candidate_splits = detect_multi_question(submitted_text)
        return candidate_splits


def process_query_with_decomposition(
        query: str,
        retriever,
        provider: str = "claude",
        model: str = None,
        api_key: str = None,
        max_subqs: int = 3,
        persist_directory: str = "RAG/ProcessedDocuments/chroma_db"
) -> Dict[str, Any]:
    """
    Main orchestration function: detect multi-questions, decompose if needed,
    and route through the existing adaptive pipeline.

    Args:
        query: User's input query
        retriever: The retrieval system instance
        provider: LLM provider
        model: LLM model
        api_key: LLM API key
        max_subqs: Maximum sub-questions to process
        persist_directory: ChromaDB directory

    Returns:
        Dict with format:
        {
            "mode": "single" | "multi",
            "items": [
                {
                    "question": str,
                    "result": dict  # result from adaptive_rag_with_validation
                }
            ]
        }
    """

    # Build LLM once upfront
    llm = build_llm(provider=provider, model=model, api_key=api_key)

    # Step 0: Check if decomposition is needed
    needs_decompose, rough_parts = detect_multi_question(query, max_parts=max_subqs)

    if not needs_decompose:
        print(f"[Decompose] Single question detected, processing normally")

        # Run full adaptive pipeline for each sub-question
        candidates = retriever.step1_wide_retrieval(query)
        reranked = retriever.step2_rerank(candidates, query)
        final_docs = retriever.step3_adaptive_selection(reranked, query)

        result = adaptive_rag_with_validation(
            query=query,
            final_docs=final_docs,
            llm=llm,
            persist_directory=persist_directory,
        )

        return {
            "mode": "single",
            "items": [{
                "question": query,
                "result": result
            }]
        }

    # Multi-question detected: use LLM to clean up splits
    print(f"[Decompose] Multi-question detected, decomposing...")
    sub_questions = decompose_to_json(query, llm)
    sub_questions = sub_questions[:max_subqs]  # Cap to max

    print(f"[Decompose] Processing {len(sub_questions)} sub-questions")

    items = []

    for i, sub_query in enumerate(sub_questions, 1):
        print(f"[Decompose] Processing Q{i}: {sub_query[:100]}...")

        # Run full adaptive pipeline for each sub-question
        candidates = retriever.step1_wide_retrieval(sub_query)
        reranked = retriever.step2_rerank(candidates, sub_query)
        final_docs = retriever.step3_adaptive_selection(reranked, sub_query)

        result = adaptive_rag_with_validation(
            query=sub_query,
            final_docs=final_docs,
            llm=llm,
            persist_directory=persist_directory,
        )

        items.append({
            "question": sub_query,
            "result": result
        })

    return {
        "mode": "multi",
        "items": items
    }


def print_decomposed_results(payload: Dict[str, Any]) -> None:
    """
    Helper function to print results in a nice format.
    """
    if payload["mode"] == "single":
        print("\n================= ANSWER =================")
        print(payload["items"][0]["result"]["answer"])
    else:
        print(f"\n================= MULTI-QUESTION SUMMARY ({len(payload['items'])} questions) =================")
        for i, item in enumerate(payload["items"], 1):
            result = item["result"]
            status = result.get('validation_status', 'unknown')
            confidence = result.get('confidence', 'N/A')
            print(f"Q{i}: {item['question'][:80]}{'...' if len(item['question']) > 80 else ''}")
            print(f"    Status: {status} | Confidence: {confidence}")

        print("\n================= DETAILED ANSWERS =================")
        for i, item in enumerate(payload["items"], 1):
            print(f"\n--- Q{i}: {item['question']} ---")
            print(item["result"]["answer"])
            print("-" * 50)