import os
import shutil
import unittest

from RAG.KB.preprocess_data import preprocess_data


class TestPreprocessData(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "tests/test_output_processed"
        # Clean up before running
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_preprocessing_returns_file_paths(self):
        result = preprocess_data(self.test_output_dir)
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        for path in result.values():
            self.assertTrue(os.path.exists(path))

    def tearDown(self):
        # Clean up after test
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
