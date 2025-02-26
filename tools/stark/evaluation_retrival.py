import time
import ast
from typing import Any, Dict, List, Optional, Union

import torch
import pandas as pd
import numpy as np
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field
from torchmetrics.functional.retrieval import (
    retrieval_hit_rate,
    retrieval_reciprocal_rank,
    retrieval_recall,
    retrieval_precision,
    retrieval_average_precision,
    retrieval_normalized_dcg,
    retrieval_r_precision,
)

from config.config import config
from state.shared_state import shared_state


class StarkEvalInput(BaseModel):
    """Input schema for StarkQA evaluation tool."""

    query_file: str = Field(description="Path to the query embeddings parquet file")
    node_file: str = Field(description="Path to the node embeddings parquet file")
    batch_size: int = Field(
        default=256, description="Batch size for processing", ge=1, le=1000
    )
    split: str = Field(default="test-0.1", description="Data split to evaluate on")


def parse_answer_ids(answer_ids, max_node_id):
    """Helper function to parse answer_ids in different formats and filter invalid IDs"""
    try:
        # Parse the answer_ids first
        if isinstance(answer_ids, list):
            result = answer_ids
        elif isinstance(answer_ids, str):
            result = ast.literal_eval(answer_ids)
        elif isinstance(answer_ids, np.ndarray):
            result = answer_ids.tolist()
        elif isinstance(answer_ids, (int, np.int64, np.int32)):
            result = [int(answer_ids)]
        else:
            print(
                f"WARNING: Unexpected answer_ids type: {type(answer_ids)}, value: {answer_ids}"
            )
            return [0]

        # Filter out invalid IDs
        valid_ids = [aid for aid in result if aid <= max_node_id]
        if not valid_ids:  # If no valid IDs remain
            return [0]  # Default value
        return valid_ids

    except Exception as e:
        print(f"WARNING: Error parsing answer_ids: {str(e)}, value: {answer_ids}")
        return [0]


@tool(args_schema=StarkEvalInput)
def evaluate_stark_retrieval(
    query_file: str, node_file: str, batch_size: int = 256, split: str = "test-0.1"
) -> Dict[str, Any]:
    """Evaluate retrieval performance using StarkQA benchmark."""
    try:
        # Load data
        queries_df = pd.read_parquet(query_file)
        nodes_df = pd.read_parquet(node_file)

        print(f"Loaded {len(queries_df)} queries and {len(nodes_df)} nodes.")

        # Get max node ID first
        max_node_id = max(nodes_df.node_id)
        print(f"Max node ID: {max_node_id}")

        print("\nSample answer_ids before filtering:")
        print(queries_df["answer_ids"].head())

        # Parse and filter answer_ids
        queries_df["answer_ids"] = queries_df.answer_ids.apply(
            lambda x: parse_answer_ids(x, max_node_id)
        )

        print("\nSample answer_ids after filtering:")
        print(queries_df["answer_ids"].head())

        # Print statistics about filtered data
        original_answers = [len(aids) for aids in queries_df.answer_ids]
        total_original = sum(original_answers)
        filtered_answers = [len(aids) for aids in queries_df.answer_ids]
        total_filtered = sum(filtered_answers)

        print(f"\nTotal answer IDs before filtering: {total_original}")
        print(f"Total answer IDs after filtering: {total_filtered}")
        print(f"Filtered out {total_original - total_filtered} invalid IDs")

        # Set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")

        # Prepare node IDs
        candidate_ids = torch.LongTensor(nodes_df.node_id.tolist())

        # Process embeddings
        print("Processing embeddings...")
        query_embeddings_list = [
            np.array(emb) for emb in queries_df.query_embedded.values
        ]
        node_embeddings_list = [np.array(emb) for emb in nodes_df.x.values]

        print("Computing similarities...")

        # Calculate similarities for visualization
        query_embeddings = torch.tensor(
            np.stack(query_embeddings_list), dtype=torch.float32
        ).to(device)

        node_embeddings = torch.tensor(
            np.stack(node_embeddings_list), dtype=torch.float32
        ).to(device)

        # Calculate node similarities for graph
        node_similarities = (
            torch.nn.functional.cosine_similarity(
                node_embeddings.unsqueeze(1), node_embeddings.unsqueeze(0), dim=2
            )
            .cpu()
            .numpy()
        )

        # Store in shared state for visualization
        shared_state.set(
            config.StateKeys.KNOWLEDGE_GRAPH,
            {"nodes": nodes_df.to_dict("records"), "similarities": node_similarities},
        )

        similarity = torch.matmul(query_embeddings, node_embeddings.T).cpu()
        similarity = similarity.to(torch.float32)

        pred_ids = candidate_ids
        pred = similarity.t()

        # Initialize results
        eval_results = []
        metrics = [
            "mrr",
            "map",
            "rprecision",
            "recall@5",
            "recall@10",
            "recall@20",
            "recall@50",
            "recall@100",
            "hit@1",
            "hit@3",
            "hit@5",
            "hit@10",
            "hit@20",
            "hit@50",
        ]

        # Process in batches
        print(f"\nProcessing {len(queries_df)} queries in batches of {batch_size}...")

        for batch_start in range(0, len(queries_df), batch_size):
            batch_end = min(batch_start + batch_size, len(queries_df))
            batch_queries = queries_df.iloc[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(len(queries_df) + batch_size - 1)//batch_size}"
            )

            # Convert answer IDs to tensors
            answer_ids = [torch.LongTensor(aids) for aids in batch_queries.answer_ids]

            batch_results = evaluate_batch(
                candidate_ids=candidate_ids,
                pred_ids=pred_ids,
                pred=pred[:, batch_start:batch_end],
                answer_ids=answer_ids,
                metrics=metrics,
                device=device,
            )

            for i, result in enumerate(batch_results):
                result["query_id"] = batch_queries.iloc[i].id
                eval_results.append(result)

        print("Computing mean metrics...")
        mean_metrics = {}
        for metric in metrics:
            mean_metrics[metric] = float(np.mean([r[metric] for r in eval_results]))

        # Update shared state
        shared_state.set(config.StateKeys.EVALUATION_RESULTS, eval_results)
        shared_state.set(config.StateKeys.METRICS, mean_metrics)

        print("Evaluation complete.")
        return {
            "status": "success",
            "metrics": mean_metrics,
            "detailed_results": eval_results,
            "total_evaluated": len(eval_results),
            "nodes": len(nodes_df),
            "message": f"Successfully evaluated {len(eval_results)} queries",
        }

    except Exception as e:
        print(f"Error in evaluation process: {str(e)}")
        raise ToolException(f"Error in evaluation process: {str(e)}")


def evaluate_batch(
    candidate_ids: torch.Tensor,
    pred_ids: torch.Tensor,
    pred: torch.Tensor,
    answer_ids: List[torch.Tensor],
    metrics: List[str],
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """Evaluate a batch of predictions."""

    try:
        max_id = max(candidate_ids.max().item(), pred_ids.max().item())

        # Prepare prediction tensor with explicit dtype
        all_pred = torch.ones((max_id + 1, pred.shape[1]), dtype=torch.float32) * (
            pred.min() - 1
        )

        # Convert pred to float32 if needed
        pred = pred.to(torch.float32)

        # Index check for prediction tensor
        if pred_ids.max() >= all_pred.shape[0]:
            raise ValueError(
                f"Prediction ID {pred_ids.max()} exceeds tensor bounds {all_pred.shape[0]}"
            )

        all_pred[pred_ids, :] = pred

        # Index check for candidate selection
        valid_candidates = candidate_ids < all_pred.shape[0]
        if not valid_candidates.all():
            raise ValueError(f"Some candidate IDs exceed tensor bounds")

        all_pred = all_pred[candidate_ids].t().to(device)

        # Prepare ground truth tensor
        bool_gd = torch.zeros((max_id + 1, pred.shape[1]), dtype=torch.bool)

        # Index check for answer IDs
        all_answer_ids = torch.concat(answer_ids)
        if all_answer_ids.max() >= bool_gd.shape[0]:
            raise ValueError(
                f"Answer ID {all_answer_ids.max()} exceeds tensor bounds {bool_gd.shape[0]}"
            )

        bool_gd[
            all_answer_ids,
            torch.repeat_interleave(
                torch.arange(len(answer_ids)),
                torch.tensor([len(x) for x in answer_ids]),
            ),
        ] = True

        bool_gd = bool_gd[candidate_ids].t().to(device)

        # Calculate metrics
        results = []
        for i in range(len(answer_ids)):
            result = {}
            for metric in metrics:
                k = int(metric.split("@")[-1]) if "@" in metric else None

                if metric == "mrr":
                    value = retrieval_reciprocal_rank(all_pred[i], bool_gd[i])
                elif metric == "rprecision":
                    value = retrieval_r_precision(all_pred[i], bool_gd[i])
                elif "hit" in metric:
                    value = retrieval_hit_rate(all_pred[i], bool_gd[i], k)
                elif "recall" in metric:
                    value = retrieval_recall(all_pred[i], bool_gd[i], k)
                elif "precision" in metric:
                    value = retrieval_precision(all_pred[i], bool_gd[i], k)
                elif "map" in metric:
                    value = retrieval_average_precision(all_pred[i], bool_gd[i], k)
                elif "ndcg" in metric:
                    value = retrieval_normalized_dcg(all_pred[i], bool_gd[i], k)

                result[metric] = float(value)

            results.append(result)

        return results
    except Exception as e:
        raise ValueError(f"Error in evaluate_batch: {str(e)}")
