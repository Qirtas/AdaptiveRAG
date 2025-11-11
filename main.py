import argparse
import json
import logging
import os
import pickle
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever
from RAG.Evaluation.retriever_evaluation import (collect_scores,
                                                 load_questions,
                                                 summarize_and_plot)
from RAG.Evaluation.test_set_creation import (create_entity_name_typo,
                                              create_question_word_typo,
                                              generate_bsc_family_questions,
                                              generate_bsc_subfamily_questions,
                                              generate_criteria_questions,
                                              generate_kpi_questions,
                                              generate_objective_questions,
                                              generate_relationship_questions)
from RAG.Evaluation.test_set_out_of_domain import (convert_to_custom_format,
                                                   fetch_trivia_questions)
from RAG.Evaluation.tuning_params import (evaluate_grid, load_test_set,
                                          plot_precision_recall,
                                          plotFPR)
from RAG.KB.generating_embeddings import generate_embeddings
from RAG.KB.ingest_documents import (ingest_documents,
                                     load_all_csvs_as_documents)
from RAG.KB.preprocess_data import preprocess_data
from RAG.KB.vector_DB import create_vectorstore
from RAG.LLM.llm_provider import build_llm
from RAG.LLM.rag_controller import rag_with_validation
from RAG.Evaluation.evaluate_adaptive_retrieval import run_evaluation
from RAG.Retrieval.retriever import (get_retrieval_results,
                                     get_retrieval_with_threshold,
                                     setup_retriever)
from RAG.Retrieval.adaptive_validate import adaptive_rag_with_validation
from RAG.Retrieval.query_decomposition import detect_multi_question, process_query_with_decomposition, print_decomposed_results

def run_index(persist_dir: str):
    """
    Full indexing pipeline: preprocess -> ingest -> embeddings -> vector DB.
    """
    # 1) Preprocess CSVs
    output_dir = "RAG/Content/ProcessedFiles"
    processed_files = preprocess_data(output_dir)

    # 2) Ingest to LangChain Documents (+ writes all_documents.pkl)
    documents = ingest_documents()

    # 3) Generate embeddings
    generate_embeddings()

    # 4) Create/update Chroma vector DB
    _ = create_vectorstore()
    logging.info(f"Index build complete. Vector DB at: {persist_dir}")

def _unwrap_docs(items):
    """Make sure we pass a List[Document] to the LLM."""
    out = []
    for x in items:
        if hasattr(x, "page_content"):                # LangChain Document
            out.append(x)
        elif isinstance(x, tuple) and x:              # (Document, score) or similar
            out.append(x[0])
        elif isinstance(x, dict):                     # {"doc": Document, ...} or variants
            if "doc" in x and hasattr(x["doc"], "page_content"):
                out.append(x["doc"])
            elif "document" in x and hasattr(x["document"], "page_content"):
                out.append(x["document"])
            elif "item" in x and hasattr(x["item"], "page_content"):
                out.append(x["item"])
            else:
                # fallback: stringify if nothing else matches
                out.append(str(x))
        else:
            out.append(x)
    return out

if __name__ == '__main__':


    print("Interpreter:", sys.executable)
    parser = argparse.ArgumentParser(
        description="RAGForDatamite pipeline runner"
    )
    parser.add_argument(
        "--mode",
        choices=["index", "demo", "eval", "all"],
        default="demo",
        help=(
            "index = build vector DB (preprocess -> ingest -> embed -> vectorstore); "
        ),
    )
    parser.add_argument(
        "--persist_dir",
        default="RAG/ProcessedDocuments/chroma_db",
        help="Directory for Chroma vector DB."
    )

    args = parser.parse_args()

    logger = logging.getLogger("datamite")
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.mode in ("index", "all"):
        run_index(args.persist_dir)

    logging.info("Done.")

    '''
    # 1. Preprocess CSV Files to clean for JSON related issues

    output_dir = "RAG/Content/ProcessedFiles"

    try:
        processed_files = preprocess_data(output_dir)

        logger.info("\nPreprocessing Summary:")
        logger.info("=====================")
        for file_type, file_path in processed_files.items():
            logger.info(f"{file_type}: {file_path}")

        logger.info("\nTo use these files with KB.py:")
        logger.info("```python")
        logger.info("from KB import load_csv_as_documents, load_all_csvs_as_documents")
        logger.info("")
        logger.info("# Define processed file paths")
        logger.info("csv_files = {")
        for file_type, file_path in processed_files.items():
            logger.info(f"    '{file_type}': '{file_path}',")
        logger.info("}")
        logger.info("")
        logger.info("# Load all documents")
        logger.info("all_docs = load_all_csvs_as_documents(csv_files)")
        logger.info("print(f\"Loaded {len(all_docs)} total documents\")")
        logger.info("```")

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        sys.exit(1)

    
    # 2. Documents Ingestion - converting to Langchain documents

    documents = ingest_documents()

    if documents:
        logger.info("\nSample document:")
        logger.info("-" * 80)

        sample_doc = next((doc for doc in documents if doc.metadata.get("type") == "KPI"), documents[0])

        logger.info("Content:")
        logger.info(sample_doc.page_content[:500] + "..." if len(sample_doc.page_content) > 500 else sample_doc.page_content)
        logger.info("-" * 80)
        logger.info("Metadata:")
        logger.info(sample_doc.metadata)
    

    # 3. Analysing if there is a need for text chunking

    with open('RAG/ProcessedDocuments/all_documents.pkl', 'rb') as f:
        documents = pickle.load(f)

    doc_lengths = defaultdict(list)
    for doc in documents:
        doc_type = doc.metadata.get('type', 'Unknown')
        length = len(doc.page_content)
        doc_lengths[doc_type].append(length)

    for doc_type, lengths in doc_lengths.items():
        logger.info(f"\n{doc_type} Documents:")
        logger.info(f"  Count: {len(lengths)}")
        logger.info(f"  Average length: {np.mean(lengths):.1f} characters")
        logger.info(f"  Min length: {min(lengths)} characters")
        logger.info(f"  Max length: {max(lengths)} characters")

        if lengths:
            longest_idx = np.argmax(lengths)
            longest_doc = next((doc for i, doc in enumerate(documents)
                                if doc.metadata.get('type') == doc_type
                                and i == longest_idx), None)
            if longest_doc:
                logger.info(f"  Longest document name: {longest_doc.metadata.get('name', 'Unknown')}")
                logger.info(f"  First 200 chars: {longest_doc.page_content[:200]}...")

    # 4. Generating vector embeddings from Langchain documents

    generate_embeddings()

    # 5. Creating vector DB and save embeddings to Vector DB

    vectorstore = create_vectorstore()
    '''

# Adaptive Retriever

    # Loading vector store
    persist_directory = "RAG/ProcessedDocuments/chroma_db"
    model_name = "all-MiniLM-L6-v2"
    logger.info(f"Loading vector store from {persist_directory}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    # Uncomment below if commenting the above vectorDB creations steps and using an existing DB
    vectorstore = Chroma(persist_directory=persist_directory,
                         embedding_function=embedding_function)

    csv_files = {
        'KPIs': 'RAG/Content/ProcessedFiles/clean_KPIs.csv',
        'Objectives': 'RAG/Content/ProcessedFiles/clean_Objectives.csv',
        'BSC_families': 'RAG/Content/ProcessedFiles/clean_BSC_families.csv',
        'BSC_subfamilies': 'RAG/Content/ProcessedFiles/clean_BSC_subfamilies.csv',
        'Criteria': 'RAG/Content/ProcessedFiles/clean_Criteria.csv'
    }
    documents = load_all_csvs_as_documents(csv_files)

retriever = AdaptiveRetriever(
    vectorstore=vectorstore,
    k_init=30,
    pool_cap=20
)

query = "Explain the relation between Data Price and Information Frequency."

# payload = process_query_with_decomposition(
#     query=query,
#     retriever=retriever,
#     provider="claude",
#     api_key="",
#     max_subqs=3,
#     persist_directory="RAG/ProcessedDocuments/chroma_db"
# )
#
# print_decomposed_results(payload)

candidates = retriever.step1_wide_retrieval(query)
reranked = retriever.step2_rerank(candidates, query)
final_docs = retriever.step3_adaptive_selection(reranked, query)

docs_for_llm = _unwrap_docs(final_docs)
provider = "claude"
llm = build_llm(provider=provider, api_key="")

result = adaptive_rag_with_validation(
    query=query,
    final_docs=final_docs,
    llm=llm,
    persist_directory="RAG/ProcessedDocuments/chroma_db"
)

print("\n================= ANSWER =================")
print(result["answer"])
