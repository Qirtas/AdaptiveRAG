import os
import shutil
import unittest

from langchain_community.vectorstores.chroma import Chroma

from RAG.KB.vector_DB import create_vectorstore


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "tests/test_chroma_db"
        self.embeddings_file = "RAG/ProcessedDocuments/document_embeddings.pkl"

    def test_create_vectorstore_returns_chroma(self):
        vectorstore = create_vectorstore(
            embeddings_file=self.embeddings_file,
            persist_directory=self.test_output_dir
        )

        self.assertIsInstance(vectorstore, Chroma)
        self.assertGreater(vectorstore._collection.count(), 0)

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
