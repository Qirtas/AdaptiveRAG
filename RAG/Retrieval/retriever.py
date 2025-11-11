import logging
import os
from typing import Any, Dict, List, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

def setup_retriever(persist_directory="RAG/ProcessedDocuments/chroma_db",
                    model_name="all-MiniLM-L6-v2",
                    k=3):
    """
        Loads the Chroma vector store and initializes a retriever for querying.

        Args:
            persist_directory (str): Path to the Chroma DB directory.
            model_name (str): Name of the HuggingFace embedding model.
            k (int): Number of top documents to retrieve per query.

        Returns:
            VectorStoreRetriever: A configured retriever object.
        """
    logger.info(f"Loading vector store from {persist_directory}")

    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_function)

    logger.info(f"Vector store loaded with {vectorstore._collection.count()} documents")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    logger.info(f"Retriever initialized to fetch top {k} documents")
    return retriever


def get_retrieval_results(query, retriever=None, persist_directory="RAG/ProcessedDocuments/chroma_db", k=3):
    """
    Runs a query against the vector store KB and prints similarity scores.

    Args:
        query (str): The query text.
        retriever (VectorStoreRetriever, optional): A retriever object to use. If None, creates one.
        persist_directory (str): Path to the Chroma DB directory. Used if retriever is None.
        k (int): Number of top documents to retrieve.

    Returns:
        list: List of tuples (Document, similarity_score).
    """
    if retriever is None:
        retriever = setup_retriever(persist_directory=persist_directory, k=k)

    logger.info(f"\nQuery: {query}")

    docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=k)

    logger.info(f"Retrieved {len(docs_with_scores)} documents with similarity scores")

    results = []
    for i, (doc, score) in enumerate(docs_with_scores):
        raw_cosine_similarity = 1 - score

        logger.info(f"\n--- Document {i + 1} ---")
        logger.info(f"Cosine Similarity Score: {raw_cosine_similarity:.4f} (Distance: {score:.4f})")
        logger.info(f"Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"Type: {doc.metadata.get('type', 'Unknown')}")
        logger.info(f"Name: {doc.metadata.get('name', 'Unknown')}")

        if raw_cosine_similarity > 0.48:  # Distance < 1.1
            quality = "Excellent - Direct match"
        elif raw_cosine_similarity > 0.45:  # Distance < 1.2
            quality = "Good - Highly relevant"
        elif raw_cosine_similarity > 0.40:  # Distance < 1.5
            quality = "Fair - Somewhat relevant"
        elif raw_cosine_similarity > 0.35:  # Distance < 1.9
            quality = "Poor - Likely irrelevant"
        else:
            quality = "Very Poor - Not relevant"

        # logger.info(f"Quality Assessment: {quality}")

        content_snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        logger.info(f"Content snippet: {content_snippet}")

        results.append((doc, raw_cosine_similarity))

    return results


def get_retrieval_with_threshold(
        query: str,
        k: int,
        similarity_threshold: float,
        persist_directory: str = "RAG/ProcessedDocuments/chroma_db",
        model_name: str = "all-MiniLM-L6-v2"
) -> List[Dict[str, Any]]:
    """
    Retrieve documents with both k and similarity threshold constraints.

    Args:
        query: Query text
        k: Maximum number of documents to retrieve
        similarity_threshold: Minimum similarity score (0-1) for retrieval
        persist_directory: Path to ChromaDB
        model_name: Embedding model name

    Returns:
        List of dicts with document info and scores
    """
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    initial_k = min(k * 3, 20)  # Retrieve up to 3x k or 20 docs
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=initial_k)

    filtered_results = []

    for doc, distance in docs_with_scores:
        similarity_score = 1 / (1 + distance)

        print(f"doc: {doc.metadata.get('name')}")
        print(f"similarity_score: {similarity_score}")

        if similarity_score >= similarity_threshold:
            result = {
                'document': doc,
                'similarity_score': similarity_score,
                'distance': distance,
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            filtered_results.append(result)

    filtered_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    final_results = filtered_results[:k]

    logger.info(f"Query: {query}")
    logger.info(f"Retrieved {len(docs_with_scores)} initial docs, "
                f"{len(filtered_results)} passed threshold, "
                f"returning top {len(final_results)}")

    return final_results
