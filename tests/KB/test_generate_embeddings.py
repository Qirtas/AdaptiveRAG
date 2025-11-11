import os
import unittest

from RAG.KB.generating_embeddings import generate_embeddings


class TestGenerateEmbeddings(unittest.TestCase):
    def test_generate_embeddings_output_structure(self):
        result = generate_embeddings(
            model_name="all-MiniLM-L6-v2",
            input_file="RAG/ProcessedDocuments/all_documents.pkl",
            output_file="tests/test_output_embeddings.pkl"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("documents", result)
        self.assertIn("embeddings", result)
        self.assertEqual(len(result["documents"]), len(result["embeddings"]))

    def tearDown(self):
        # deleting the test embedding file
        path = "tests/test_output_embeddings.pkl"
        if os.path.exists(path):
            os.remove(path)


