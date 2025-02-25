import os
import sys
import unittest
from typing import Any, Dict, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pandas as pd
import torch
import numpy as np
from agents.main_agent import main_agent
from agents.stark_agent import stark_agent
from config.config import config
from langchain_core.messages import HumanMessage
from state.shared_state import shared_state
from tools.stark.evaluation_retrival import evaluate_stark_retrieval


class TestStarkEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup test data paths
        cls.test_dir = "test_data"
        if not os.path.exists(cls.test_dir):
            os.makedirs(cls.test_dir)

        # Create sample test data
        cls.create_test_data()

    @classmethod
    def tearDownClass(cls):
        # Cleanup test data
        if os.path.exists(cls.test_dir):
            import shutil

            shutil.rmtree(cls.test_dir)

    @classmethod
    def create_test_data(cls):
        # Create sample query embeddings
        queries_data = {
            "id": [0, 1],
            "query": ["test query 1", "test query 2"],
            "answer_ids": [[1], [2]],  # Simple list instead of numpy array
            "query_embedded": [torch.randn(768).numpy().tolist() for _ in range(2)],
        }
        queries_df = pd.DataFrame(queries_data)
        queries_df.to_parquet(os.path.join(cls.test_dir, "test_queries.parquet"))

        # Create sample node embeddings
        nodes_data = {
            "node_id": [1, 2, 3],
            "node_name": ["node1", "node2", "node3"],
            "node_type": ["type1", "type2", "type1"],
            "x": [torch.randn(768).numpy().tolist() for _ in range(3)],
        }
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_parquet(os.path.join(cls.test_dir, "test_nodes.parquet"))

    def test_evaluation_tool(self):
        """Test the StarkQA evaluation tool"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file": os.path.join(self.test_dir, "test_queries.parquet"),
                "node_file": os.path.join(self.test_dir, "test_nodes.parquet"),
                "batch_size": 2,
                "split": "test",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("metrics", result)
        self.assertIn("detailed_results", result)

    def test_stark_agent(self):
        """Test the StarkQA agent"""
        query = "Evaluate the retrieval performance on these embeddings"
        state = {"messages": [{"role": "user", "content": query}]}

        result = stark_agent.invoke(state)
        self.assertIn("messages", result)

    def test_main_agent_routing(self):
        """Test main agent routing to StarkQA agent"""
        query = "Can you evaluate the retrieval performance of this model?"
        state = {"messages": [HumanMessage(content=query)]}

        result = main_agent.invoke(state)

        # Check if routed to stark agent
        current_agent = shared_state.get(config.StateKeys.CURRENT_AGENT)
        self.assertEqual(current_agent, config.AgentNames.STARK)

    def test_shared_state_updates(self):
        """Test shared state updates during evaluation"""
        result = evaluate_stark_retrieval.invoke(
            {
                "query_file": os.path.join(self.test_dir, "test_queries.parquet"),
                "node_file": os.path.join(self.test_dir, "test_nodes.parquet"),
                "batch_size": 2,
                "split": "test",
            }
        )

        # Check state updates
        self.assertIsNotNone(shared_state.get(config.StateKeys.EVALUATION_RESULTS))
        self.assertIsNotNone(shared_state.get(config.StateKeys.METRICS))


if __name__ == "__main__":
    unittest.main()
