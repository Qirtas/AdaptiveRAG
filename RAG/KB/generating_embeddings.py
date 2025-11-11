import os
import pickle

#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings



def generate_embeddings(model_name="all-MiniLM-L6-v2", input_file="RAG/ProcessedDocuments/all_documents.pkl",
                      output_file="RAG/ProcessedDocuments/document_embeddings.pkl"):
    """
        Generates embeddings for LangChain Documents using HuggingFace model and saves to a pickle file.

        Args:
            model_name (str): Name of the HuggingFace embedding model.
            input_file (str): Path to input pickle containing LangChain Documents.
            output_file (str): Path to save the output embeddings pickle.

        Returns:
            dict: Dictionary containing documents, embeddings, and model name.
        """

    print(f"Loading documents from {input_file}")

    with open(input_file, 'rb') as f:
        documents = pickle.load(f)

    print(f"Loaded {len(documents)} documents")

    print(f"Loading embedding model: {model_name}")
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

    texts = [doc.page_content for doc in documents]

    print("Generating embeddings...")
    embeddings = embeddings_model.embed_documents(texts)
    print(embeddings[2])
    print(embeddings[20])
    print(embeddings[250])

    document_embeddings = {
        "documents": documents,
        "embeddings": embeddings,
        "model_name": model_name
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(document_embeddings, f)

    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    print(f"Saved embeddings to {output_file}")

    return document_embeddings
