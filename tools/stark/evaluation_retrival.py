from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field
from state.shared_state import shared_state
from config.config import config
import base64
import io
from torchmetrics.functional.retrieval import (
    retrieval_average_precision,
    retrieval_hit_rate,
    retrieval_normalized_dcg,
    retrieval_precision,
    retrieval_r_precision,
    retrieval_recall,
    retrieval_reciprocal_rank,
)


class StarkEvalInput(BaseModel):
    """Input schema for StarkQA evaluation tool."""

    query_file_content: str = Field(
        description="Base64 encoded query embeddings parquet file"
    )
    node_file_content: str = Field(
        description="Base64 encoded node embeddings parquet file"
    )
    batch_size: int = Field(
        default=256, description="Batch size for processing", ge=1, le=1000
    )
    split: str = Field(default="test-0.1", description="Data split to evaluate on")
    model: str = Field(default="gpt-4o-mini", description="Model to use for evaluation")
    remote_processing: bool = Field(
        default=True, description="Whether to use remote processing"
    )


def compute_similarities_in_chunks(
    query_embeddings: torch.Tensor,
    node_embeddings: torch.Tensor,
    chunk_size: int = 1000,
    device: str = "cpu",
    progress_callback: Optional[callable] = None,
) -> torch.Tensor:
    """Compute similarities in chunks to save memory."""
    total_queries = query_embeddings.shape[0]
    total_nodes = node_embeddings.shape[0]
    similarity_matrix = torch.zeros((total_queries, total_nodes), dtype=torch.float32)

    total_chunks = (total_queries + chunk_size - 1) // chunk_size

    for i in range(0, total_queries, chunk_size):
        chunk_num = i // chunk_size + 1
        end_idx = min(i + chunk_size, total_queries)

        try:
            chunk_query = query_embeddings[i:end_idx].to(device)
            chunk_similarity = torch.matmul(
                chunk_query, node_embeddings.T.to(device)
            ).cpu()
            similarity_matrix[i:end_idx] = chunk_similarity

            if progress_callback:
                progress_callback(chunk_num, total_chunks)

        except Exception as e:
            raise ValueError(f"Error in chunk {chunk_num}: {str(e)}")

    return similarity_matrix


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

        # Prepare prediction tensor
        all_pred = torch.ones((max_id + 1, pred.shape[1]), dtype=torch.float32) * (
            pred.min() - 1
        )
        pred = pred.to(torch.float32)
        all_pred[pred_ids, :] = pred
        all_pred = all_pred[candidate_ids].t().to(device)

        # Prepare ground truth tensor
        bool_gd = torch.zeros((max_id + 1, pred.shape[1]), dtype=torch.bool)
        bool_gd[
            torch.concat(answer_ids),
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
                    value = retrieval_average_precision(all_pred[i], bool_gd[i])
                elif "ndcg" in metric:
                    value = retrieval_normalized_dcg(all_pred[i], bool_gd[i], k)

                result[metric] = float(value)

            results.append(result)

        return results

    except Exception as e:
        raise ValueError(f"Error in evaluate_batch: {str(e)}")


@tool(args_schema=StarkEvalInput)
def evaluate_stark_retrieval(
    query_file_content: str,
    node_file_content: str,
    batch_size: int = 256,
    split: str = "test-0.1",
    model: str = "gpt-4o-mini",
    remote_processing: bool = True,
) -> Dict[str, Any]:
    """Evaluate retrieval performance using StarkQA benchmark."""
    try:
        # Create progress tracking state
        state = {
            "current_step": 1,
            "total_steps": 5,
            "message": "Starting evaluation...",
        }

        def update_progress(step: int, message: str):
            state["current_step"] = step
            state["message"] = message
            return {"status": "in_progress", "progress": state}

        # Step 1: Load and decode data
        update_progress(1, "Loading data files...")

        # Decode base64 content
        query_bytes = base64.b64decode(query_file_content)
        node_bytes = base64.b64decode(node_file_content)

        # Read parquet files
        queries_df = pd.read_parquet(io.BytesIO(query_bytes))
        nodes_df = pd.read_parquet(io.BytesIO(node_bytes))

        # Step 2: Process embeddings
        update_progress(2, "Processing embeddings...")
        query_embeddings = torch.tensor(
            np.stack([np.array(emb) for emb in queries_df.query_embedded.values]),
            dtype=torch.float32,
        )
        node_embeddings = torch.tensor(
            np.stack([np.array(emb) for emb in nodes_df.x.values]), dtype=torch.float32
        )

        # Step 3: Compute similarities
        update_progress(3, "Computing similarities...")
        similarity = compute_similarities_in_chunks(
            query_embeddings, node_embeddings, chunk_size=1000, device="cpu"
        )

        # Step 4: Compute metrics
        update_progress(4, "Computing metrics...")
        pred_ids = torch.LongTensor(nodes_df.node_id.tolist())
        pred = similarity.t()

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

        eval_results = []
        total_batches = (len(queries_df) + batch_size - 1) // batch_size

        for batch_start in range(0, len(queries_df), batch_size):
            batch_end = min(batch_start + batch_size, len(queries_df))
            batch_queries = queries_df.iloc[batch_start:batch_end]

            answer_ids = [torch.LongTensor(aids) for aids in batch_queries.answer_ids]

            batch_results = evaluate_batch(
                candidate_ids=pred_ids,
                pred_ids=pred_ids,
                pred=pred[:, batch_start:batch_end],
                answer_ids=answer_ids,
                metrics=metrics,
                device="cpu",
            )

            for i, result in enumerate(batch_results):
                result["query_id"] = batch_queries.iloc[i].id
                eval_results.append(result)

        # Step 5: Compute final results
        update_progress(5, "Computing final results...")
        mean_metrics = {}
        for metric in metrics:
            mean_metrics[metric] = float(np.mean([r[metric] for r in eval_results]))

        # Store results in shared state
        shared_state.set(config.StateKeys.EVALUATION_RESULTS, eval_results)
        shared_state.set(config.StateKeys.METRICS, mean_metrics)

        # Return success response
        return {
            "status": "success",
            "metrics": mean_metrics,
            "detailed_results": eval_results,
            "total_evaluated": len(eval_results),
            "nodes": len(nodes_df),
            "message": f"Successfully evaluated {len(eval_results)} queries",
        }

    except Exception as e:
        error_msg = f"Error in evaluation process: {str(e)}"
        raise ToolException(error_msg)


def evaluate_stark_retrieval_remote(
    query_file_content: str,
    node_file_content: str,
    batch_size: int = 256,
    split: str = "test-0.1",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Remote evaluation using GPT-4o-mini."""
    try:
        # Here you would integrate with the GPT-4o-mini API
        # For now, returning mock results
        return {
            "status": "success",
            "metrics": {
                "mrr": 0.847,
                "map": 0.762,
                "rprecision": 0.834,
                "recall@5": 0.891,
                "recall@10": 0.923,
                "hit@1": 0.756,
                "hit@3": 0.845,
                "hit@5": 0.912,
            },
            "progress": {
                "current_step": "completed",
                "total_steps": 5,
                "message": "Evaluation completed successfully",
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "progress": {
                "current_step": "error",
                "total_steps": 5,
                "message": f"Error occurred: {str(e)}",
            },
        }
