import os

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from RAG.Retrieval.retriever import get_retrieval_results


class ValidationSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with the same embedding model which we used in the vector store"""
        self.embedding_model = SentenceTransformer(model_name)

    def get_document_set_embedding(self, docs):
        """
        Creating a single embedding representing the semantic content of multiple documents
        """
        # Combine all document content
        combined_content = "\n".join([doc.page_content for doc in docs])

        # Get embedding for combined content
        embedding = self.embedding_model.encode([combined_content])
        return embedding[0]

    def get_individual_document_embeddings(self, docs):
        """
        Get individual embeddings for each document
        """
        contents = [doc.page_content for doc in docs]
        embeddings = self.embedding_model.encode(contents)
        return embeddings

    def calculate_semantic_overlap_v1(self, original_docs, validation_docs):
        """
        Method 1: Compare combined document set embeddings to check if the document sets discuss similar topics overall
        We using this specifically to detect topic drift (If LLM completely changes subject)
        """
        # Get embeddings for document sets
        original_embedding = self.get_document_set_embedding(original_docs)
        validation_embedding = self.get_document_set_embedding(validation_docs)

        # Calculate cosine similarity
        similarity = cosine_similarity(
            original_embedding.reshape(1, -1),
            validation_embedding.reshape(1, -1)
        )[0][0]

        return similarity


    def validation_analysis(self, original_docs, validation_docs):
        """
        Performing validation using multiple methods
        """
        set_similarity = self.calculate_semantic_overlap_v1(original_docs, validation_docs)

        combined_score = set_similarity

        return {
            "set_similarity": set_similarity,
            "avg_max_similarity": "",
            "jaccard_similarity": "",
            "combined_score": combined_score,
            "individual_scores": {
                "document_set": set_similarity,
                "average_maximum": "",
                "jaccard_style": ""
            }
        }


def extract_key_concepts_from_answer(answer):
    """
    Key term extraction using the embedding model
    """
    # Removing common words and extract meaningful phrases
    import re

    cleaned = re.sub(r'[^\w\s]', ' ', answer.lower())

    words = cleaned.split()
    stop_words = {
        'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those',
        'it', 'they', 'them', 'their', 'there', 'then', 'than', 'when',
        'where', 'how', 'what', 'who', 'which', 'why', 'can', 'could',
        'would', 'should', 'will', 'may', 'might', 'must', 'shall'
    }

    meaningful_terms = [
        word for word in words
        if len(word) > 3 and word not in stop_words
    ]

    validation_query = ' '.join(meaningful_terms[:20])

    return validation_query


def validation_search(original_question, llm_answer, original_docs,
                               persist_directory="RAG/ProcessedDocuments/chroma_db"):
    """
    Validation search using semantic similarity
    """
    print(f"\n{'=' * 60}")
    print("PERFORMING VALIDATION SEARCH")
    print(f"{'=' * 60}")

    validator = ValidationSearch()

    # Step 1: Extract key concepts from LLM answer
    validation_query = extract_key_concepts_from_answer(llm_answer)
    print(f"Validation query: '{validation_query}'")

    # Step 2: Search knowledge base with validation query
    validation_results = get_retrieval_results(
        validation_query,
        persist_directory=persist_directory,
        k=len(original_docs)  # Get same number as original
    )
    validation_docs = [doc for doc, score in validation_results]

    # Step 3: do semantic analysis
    analysis = validator.validation_analysis(original_docs, validation_docs)

    # Step 4: Determine validation status based on combined score
    combined_score = analysis["combined_score"]

    if combined_score >= 0.60:  # High semantic matching
        validation_status = "PASSED"
        confidence = "HIGH"
    elif combined_score >= 0.50:  # Medium semantic matching
        validation_status = "PARTIAL"
        confidence = "MEDIUM"
    else:  # Low semantic matching
        validation_status = "FAILED"
        confidence = "LOW"

    print(f"Validation Analysis:")
    print(f"Document Set Similarity: {analysis['set_similarity']:.3f}")

    print(f"\n Document Analysis:")
    print(f"Original documents: {len(original_docs)}")
    print(f"Validation documents: {len(validation_docs)}")

    original_names = [doc.metadata.get('name', 'Unknown') for doc in original_docs]
    validation_names = [doc.metadata.get('name', 'Unknown') for doc in validation_docs]

    print(f"Original docs: {original_names}")
    print(f"Validation docs: {validation_names}")

    return {
        "validation_query": validation_query,
        "validation_docs": validation_docs,
        "semantic_analysis": analysis,
        "combined_score": combined_score,
        "validation_status": validation_status,
        "confidence": confidence,
        "detailed_scores": analysis["individual_scores"]
    }