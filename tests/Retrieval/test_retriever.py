import unittest

from langchain.vectorstores.base import VectorStoreRetriever

from RAG.Retrieval.retriever import (run_test_queries, setup_retriever,
                                     test_retrieval)


class TestRetriever(unittest.TestCase):
    def test_setup_retriever_returns_valid_object(self):
        retriever = setup_retriever(k=2)
        self.assertIsInstance(retriever, VectorStoreRetriever)

    def test_test_retrieval_returns_documents(self):
        query = "What is Access Cost?"
        docs = test_retrieval(query, k=2)
        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)

    def test_run_test_queries_returns_dict(self):
        queries = ["Explain the Financial Perspective in BSC"]
        results = run_test_queries(queries, k=2)
        self.assertIsInstance(results, dict)
        self.assertIn(queries[0], results)
        self.assertIsInstance(results[queries[0]], list)
