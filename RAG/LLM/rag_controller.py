from typing import Any, Dict, List, Optional
from RAG.LLM.validation_search import validation_search
from RAG.LLM.llm_provider import LLMClient
from RAG.Retrieval.retriever import get_retrieval_results


def rag_with_validation(query, min_similarity=0.40,
                                 persist_directory="RAG/ProcessedDocuments/chroma_db", k=3, retriever=None,
                        llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    """
    RAG system with semantic validation step
    """
    print(f"\n{'=' * 80}")
    print(f"Query: {query}")
    print(f"{'=' * 80}")

    # Step 1: Initial retrieval from KB and quality control
    results = get_retrieval_results(query, retriever, persist_directory=persist_directory, k=k)

    if not results or results[0][1] < min_similarity:
        best_similarity = results[0][1] if results else 0
        print(f"Best similarityyy ({best_similarity:.4f}) below threshold ({min_similarity})")
        return {
            "answer": "I don't have relevant information to answer this question.",
            "sources": [],
            "quality_check": "failed",
            "validation": None
        }

    # Step 2: Filter documents and generate answer using injected LLM
    filtered_docs = [result[0] for result in results if result[1] >= min_similarity]
    print(f"Found {len(filtered_docs)} documents above similarity threshold ({min_similarity})")

    if llm is None:
        raise RuntimeError("LLM client not provided to rag_with_validation")

    llm_result = llm.ask_with_docs(query, filtered_docs)
    print(f"[RAG] LLM Result Before Validation Search: {llm_result['answer'][:120]}...")

    # Step 3: Perform validation search
    validation_result = validation_search(
        original_question=query,
        llm_answer=llm_result["answer"],
        original_docs=filtered_docs,
        persist_directory=persist_directory
    )

    # Step 4: Combine results
    final_result = {
        "answer": llm_result["answer"],
        "sources": llm_result["sources"],
        "quality_check": "passed",
        "validation": validation_result,
        "confidence": validation_result["confidence"],
        "validation_status": validation_result["validation_status"],
        "semantic_scores": validation_result["detailed_scores"]
    }

    return final_result