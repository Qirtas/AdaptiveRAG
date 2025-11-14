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
import yaml
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from RAG.Retrieval.adaptive_retriever import AdaptiveRetriever
from RAG.Retrieval.spell_checker import process_user_query
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
    Returns the created vectorstore.
    """
    # 1) Preprocess CSVs
    output_dir = "RAG/Content/ProcessedFiles"
    processed_files = preprocess_data(output_dir)

    # 2) Ingest to LangChain Documents (+ writes all_documents.pkl)
    documents = ingest_documents()

    # 3) Generate embeddings
    generate_embeddings()

    # 4) Create/update Chroma vector DB and return it
    vectorstore = create_vectorstore()
    logging.info(f"Index build complete. Vector DB at: {persist_dir}")

    return vectorstore


def vector_db_exists(persist_dir: str) -> bool:
    """Check if vector DB exists and has data."""
    if not os.path.exists(persist_dir):
        return False

    # Check if directory has content
    try:
        contents = os.listdir(persist_dir)
        return len(contents) > 0
    except Exception:
        return False


def setup_vector_store(config: dict):
    """
    Automatically setup vector store:
    - If exists and force_rebuild=False: load existing
    - Otherwise: build from scratch
    """
    vs_config = config['vector_store']
    persist_directory = vs_config['persist_directory']
    model_name = vs_config['model_name']
    force_rebuild = vs_config.get('force_rebuild', False)

    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    # Check if we need to build or can load existing
    if vector_db_exists(persist_directory) and not force_rebuild:
        logger.info(f"Loading existing vector store from {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function
        )
    else:
        if force_rebuild:
            logger.info("Force rebuild enabled. Rebuilding vector store...")
        else:
            logger.info("Vector store not found. Building from scratch...")

        # Run full indexing pipeline and get the vectorstore directly
        vectorstore = run_index(persist_directory)

    return vectorstore


if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("Interpreter:", sys.executable)
    parser = argparse.ArgumentParser(
        description="RAGForDatamite pipeline runner"
    )
    parser.add_argument(
        "--mode",
        choices=["index", "demo", "eval", "all"],
        default="demo",
        help="index = force rebuild vector DB; demo = run query demo"
    )
    parser.add_argument(
        "--persist_dir",
        default=None,
        help="Directory for Chroma vector DB (overrides config)"
    )

    args = parser.parse_args()

    logger = logging.getLogger("datamite")
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Override config if persist_dir provided via CLI
    if args.persist_dir:
        config['vector_store']['persist_directory'] = args.persist_dir

    # If mode is "index", force rebuild
    if args.mode in ("index", "all"):
        config['vector_store']['force_rebuild'] = True

    # ============================================================
    # AUTOMATIC VECTOR STORE SETUP
    # ============================================================
    vectorstore = setup_vector_store(config)

    # Load CSV documents for retriever
    csv_files = config['csv_files']
    documents = load_all_csvs_as_documents(csv_files)

    # ============================================================
    # SETUP ADAPTIVE RETRIEVER
    # ============================================================
    retriever_config = config['retriever']
    retriever = AdaptiveRetriever(
        vectorstore=vectorstore,
        k_init=retriever_config['k_init'],
        pool_cap=retriever_config['pool_cap']
    )

    # ============================================================
    # PROCESS QUERY
    # ============================================================
    question = "How does Customer Perspective vs Financial Perspective differ in BSC?"

    # Apply spell checking
    spell_config = config['spell_check']
    if spell_config['enabled']:
        query_result = process_user_query(
            query=question,
            use_variants=spell_config['use_variants'],
            custom_terms_path=spell_config['custom_terms_path'],
            confidence_threshold=spell_config['confidence_threshold']
        )

        question = query_result['corrected']

        if query_result['needs_correction']:
            logger.info(f"Original: '{query_result['original']}' → Corrected: '{question}'")
            logger.info(f"Confidence: {query_result['confidence']:.2f}")

            if spell_config['use_variants']:
                logger.info(f"Variants: {query_result['variants']}")
    else:
        logger.info("Spell checking is disabled")

    print(f"\nFinal query: {question}\n")

    # Get LLM configuration
    llm_config = config['llm']
    provider = llm_config['provider']
    provider_settings = llm_config[provider]

    logger.info(f"Using LLM provider: {provider}")

    # Process query with decomposition
    payload = process_query_with_decomposition(
        query=question,
        retriever=retriever,
        provider=provider,
        api_key=provider_settings['api_key'],
        max_subqs=3,
        persist_directory=config['vector_store']['persist_directory']
    )

    print_decomposed_results(payload)

    logging.info("Done.")

    
# --------------

# Manually calling each step for adaptive retriver just for testing

# candidates = retriever.step1_wide_retrieval(question)
# reranked = retriever.step2_rerank(candidates, question)
# final_docs = retriever.step3_adaptive_selection(reranked, question)
#
# docs_for_llm = _unwrap_docs(final_docs)
# llm_config = config['llm']
# provider = llm_config['provider']
# provider_settings = llm_config[provider]
#
# llm = build_llm(provider=provider, api_key=provider_settings['api_key'])
#
# result = adaptive_rag_with_validation(
#     query=question,
#     final_docs=final_docs,
#     llm=llm,
#     persist_directory="RAG/ProcessedDocuments/chroma_db"
# )
#
# print("\n================= ANSWER =================")
# print(result["answer"])
