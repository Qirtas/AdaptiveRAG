import logging
import os
import pickle

import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

def create_vectorstore(embeddings_file="RAG/ProcessedDocuments/document_embeddings.pkl",
                       persist_directory="RAG/ProcessedDocuments/chroma_db"):
    """
      Creates a Chroma vector store from document embeddings and persists it.

      Args:
          embeddings_file (str): Path to the pickle file containing documents and embeddings.
          persist_directory (str): Directory where the Chroma DB will be saved.

      Returns:
          Chroma: The Chroma vector store object.
      """

    logger.info(f"Loading document embeddings from {embeddings_file}")

    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    documents = data["documents"]
    model_name = data["model_name"]

    logger.info(f"Loaded {len(documents)} documents with embeddings from model {model_name}")

    logger.info(f"Initializing embedding model: {model_name}")
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)

    client_settings = Settings(anonymized_telemetry=False, allow_reset=True)
    chroma_client = chromadb.Client(settings=client_settings)

    logger.info(f"Creating Chroma vector store in {persist_directory}")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=HuggingFaceEmbeddings(model_name=model_name),
        persist_directory="RAG/ProcessedDocuments/chroma_db",
        client_settings=Settings(
            is_persistent=True,
            persist_directory="RAG/ProcessedDocuments/chroma_db",
            anonymized_telemetry=False,
        ),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # vectorstore.persist()

    logger.info(f"Vector store created and saved to {persist_directory}")
    logger.info(f"Total documents in vector store: {vectorstore._collection.count()}")
    logger.info(f"vector store metadata: {vectorstore._collection.metadata}")

    return vectorstore
