import time
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


@tool(args_schema=StarkEvalInput)
def evaluate_stark_retrieval(
    query_file: str, node_file: str, batch_size: int = 256, split: str = "test-0.1"
) -> Dict[str, Any]:
    """Evaluate retrieval performance using StarkQA benchmark.

    Best for:
    - Evaluating LLM retrieval performance
    - Benchmarking on semi-structured knowledge bases
    - Computing standard retrieval metrics

    Examples:
    - Computing MRR, MAP, Recall@K metrics
    - Evaluating vector similarity search
    - Analyzing knowledge graph retrieval

    Args:
        query_file: Path to query embeddings parquet file
        node_file: Path to node embeddings parquet file
        batch_size: Batch size for processing
        split: Data split to evaluate on

    Returns:
        Dict containing evaluation metrics and results
    """
    try:
        # Load data
        queries_df = pd.read_parquet(query_file)
        nodes_df = pd.read_parquet(node_file)

        # Set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Prepare node IDs
        candidate_ids = torch.LongTensor(nodes_df.node_id.tolist())

        # Define metrics
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

        # Initialize results
        eval_results = []

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                # Process in batches
                for batch_start in range(0, len(queries_df), batch_size):
                    batch_queries = queries_df.iloc[
                        batch_start : batch_start + batch_size
                    ]

                    # Calculate similarities
                    query_embeddings = torch.tensor(
                        np.stack(batch_queries.query_embedded.values)
                    ).to(device)
                    node_embeddings = torch.tensor(np.stack(nodes_df.x.values)).to(
                        device
                    )

                    similarity = torch.matmul(query_embeddings, node_embeddings.T).cpu()

                    # Prepare predictions
                    pred_ids = candidate_ids
                    pred = similarity.t()
                    answer_ids = [
                        torch.LongTensor(aids) for aids in batch_queries.answer_ids
                    ]

                    # Evaluate batch
                    batch_results = evaluate_batch(
                        candidate_ids=candidate_ids,
                        pred_ids=pred_ids,
                        pred=pred,
                        answer_ids=answer_ids,
                        metrics=metrics,
                        device=device,
                    )

                    # Add query IDs and metadata
                    for i, result in enumerate(batch_results):
                        result["query_id"] = batch_queries.iloc[i].id
                        eval_results.append(result)

                # Calculate mean metrics
                mean_metrics = {}
                for metric in metrics:
                    mean_metrics[metric] = np.mean([r[metric] for r in eval_results])

                # Update shared state
                shared_state.set(config.StateKeys.EVALUATION_RESULTS, eval_results)
                shared_state.set(config.StateKeys.METRICS, mean_metrics)

                return {
                    "status": "success",
                    "metrics": mean_metrics,
                    "detailed_results": eval_results,
                    "total_evaluated": len(eval_results),
                    "message": f"Successfully evaluated {len(eval_results)} queries",
                }

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise ToolException(f"Error during evaluation: {str(e)}")

        raise ToolException(f"Failed after {max_retries} attempts")

    except Exception as e:
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

    # Prepare prediction tensor
    all_pred = torch.ones(
        (max(candidate_ids) + 1, pred.shape[1]), dtype=torch.float
    ) * (pred.min() - 1)
    all_pred[pred_ids, :] = pred
    all_pred = all_pred[candidate_ids].t().to(device)

    # Prepare ground truth tensor
    bool_gd = torch.zeros((max(candidate_ids) + 1, pred.shape[1]), dtype=torch.bool)
    bool_gd[
        torch.concat(answer_ids),
        torch.repeat_interleave(
            torch.arange(len(answer_ids)), torch.tensor([len(x) for x in answer_ids])
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
