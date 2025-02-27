import base64
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

# Add your project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from tools.stark.evaluation_retrival import evaluate_stark_retrieval


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        """Create sample data for testing"""
        # Create sample query embeddings
        self.queries_data = pd.DataFrame(
            {
                "id": range(5),
                "query": [f"query_{i}" for i in range(5)],
                "answer_ids": [[i] for i in range(5)],
                "query_embedded": [torch.randn(384).tolist() for _ in range(5)],
            }
        )

        # Create sample node embeddings
        self.nodes_data = pd.DataFrame(
            {"node_id": range(10), "x": [torch.randn(384).tolist() for _ in range(10)]}
        )

        # Save to temporary parquet files
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            self.queries_data.to_parquet(f.name)
            self.query_file = f.name
            with open(f.name, "rb") as f2:
                self.query_content = base64.b64encode(f2.read()).decode()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            self.nodes_data.to_parquet(f.name)
            self.node_file = f.name
            with open(f.name, "rb") as f2:
                self.node_content = base64.b64encode(f2.read()).decode()

    def tearDown(self):
        """Clean up temporary files"""
        try:
            os.unlink(self.query_file)
            os.unlink(self.node_file)
        except:
            pass

    def test_local_evaluation(self):
        """Test local evaluation"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file_content": self.query_content,
                "node_file_content": self.node_content,
                "batch_size": 2,
                "split": "test",
                "remote_processing": False,
            }
        )

        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertIn("metrics", result)
        self.assertIn("detailed_results", result)

        # Verify metrics
        metrics = result["metrics"]
        for metric in ["mrr", "map", "rprecision"]:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)

    def test_remote_evaluation(self):
        """Test remote evaluation"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file_content": self.query_content,
                "node_file_content": self.node_content,
                "batch_size": 2,
                "split": "test",
                "model": "gpt-4o-mini",
                "remote_processing": True,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("metrics", result)

    def test_invalid_input(self):
        """Test handling of invalid input"""
        with self.assertRaises(Exception):
            evaluate_stark_retrieval.invoke(
                {
                    "query_file_content": "invalid",
                    "node_file_content": "invalid",
                    "batch_size": 2,
                    "split": "test",
                }
            )

    def test_batch_processing(self):
        """Test different batch sizes"""
        batch_sizes = [1, 2, 5]
        for batch_size in batch_sizes:
            result = evaluate_stark_retrieval.invoke(
                {
                    "query_file_content": self.query_content,
                    "node_file_content": self.node_content,
                    "batch_size": batch_size,
                    "split": "test",
                }
            )
            self.assertEqual(result["status"], "success")

    def test_progress_tracking(self):
        """Test progress tracking functionality"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file_content": self.query_content,
                "node_file_content": self.node_content,
                "batch_size": 2,
                "split": "test",
            }
        )

        # Verify result contains progress information
        self.assertEqual(result["status"], "success")
        self.assertGreater(len(result["detailed_results"]), 0)

    def test_metric_ranges(self):
        """Test if metrics are within valid ranges"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file_content": self.query_content,
                "node_file_content": self.node_content,
                "batch_size": 2,
                "split": "test",
            }
        )

        metrics = result["metrics"]
        for metric_name, value in metrics.items():
            self.assertGreaterEqual(value, 0.0, f"{metric_name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{metric_name} should be <= 1")

    def test_batch_size_validation(self):
        """Test batch size validation"""
        invalid_batch_sizes = [-1, 0, 1001]
        for batch_size in invalid_batch_sizes:
            with self.assertRaises(Exception):
                evaluate_stark_retrieval.invoke(
                    {
                        "query_file_content": self.query_content,
                        "node_file_content": self.node_content,
                        "batch_size": batch_size,
                        "split": "test",
                    }
                )


if __name__ == "__main__":
    unittest.main()
