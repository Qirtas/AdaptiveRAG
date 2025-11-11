import unittest

from langchain.docstore.document import Document

from RAG.KB.ingest_documents import ingest_documents


class TestIngestDocuments(unittest.TestCase):
    def test_ingestion_returns_documents(self):
        documents = ingest_documents()
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        self.assertTrue(all(isinstance(doc, Document) for doc in documents))

