from typing import Any, Dict, List, Optional
from RAG.LLM.llm_provider import LLMClient
from RAG.LLM.validation_search import validation_search

def _unwrap_docs(items) -> List[Any]:
    out=[]
    for x in items:
        if hasattr(x, "page_content"):
            out.append(x)
        elif isinstance(x, tuple) and x:
            out.append(x[0])
        elif isinstance(x, dict):
            for key in ("doc", "document", "item"):
                if key in x and hasattr(x[key], "page_content"):
                    out.append(x[key]); break
            else:
                out.append(str(x))
        else:
            out.append(x)
    return out

def adaptive_rag_with_validation(
    query: str,
    final_docs: List[Any],
    llm: LLMClient,
    persist_directory: str = "RAG/ProcessedDocuments/chroma_db",
) -> Dict[str, Any]:
    """
    Use adaptive-selected docs -> ask LLM -> run validation_search -> package results.
    """
    docs_for_llm = _unwrap_docs(final_docs)

    # 1) Ask LLM using only the adaptive docs
    llm_result = llm.ask_with_docs(query, docs_for_llm)
    answer = llm_result["answer"]

    # 2) Validate the LLM answer against the KB
    validation_result = validation_search(
        original_question=query,
        llm_answer=answer,
        original_docs=docs_for_llm,
        persist_directory=persist_directory,
    )

    # 3) Assemble a consistent response payload
    return {
        "answer": answer,
        "sources": llm_result.get("sources", docs_for_llm),
        "quality_check": "passed",
        "validation": validation_result,
        "confidence": validation_result.get("confidence"),
        "validation_status": validation_result.get("validation_status"),
        "semantic_scores": validation_result.get("detailed_scores"),
        "meta": {
            "n_context_docs": len(docs_for_llm),
        },
    }
