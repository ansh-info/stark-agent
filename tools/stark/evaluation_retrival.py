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


def compute_similarities_in_chunks(
    query_embeddings: torch.Tensor,
    node_embeddings: torch.Tensor,
    chunk_size: int = 1000,
    device: str = "cpu"
) -> torch.Tensor:
    """Compute similarities in chunks to save memory."""
    total_queries = query_embeddings.shape[0]
    total_nodes = node_embeddings.shape[0]
    similarity_matrix = torch.zeros((total_queries, total_nodes), dtype=torch.float32)
    
    total_chunks = (total_queries + chunk_size - 1) // chunk_size
    print(f"Processing {total_chunks} chunks of size {chunk_size}")
    
    for i in range(0, total_queries, chunk_size):
        chunk_num = i // chunk_size + 1
        end_idx = min(i + chunk_size, total_queries)
        print(f"Processing chunk {chunk_num}/{total_chunks} (queries {i} to {end_idx})")
        
        try:
            chunk_query = query_embeddings[i:end_idx].to(device)
            chunk_similarity = torch.matmul(chunk_query, node_embeddings.T.to(device)).cpu()
            similarity_matrix[i:end_idx] = chunk_similarity
            
            print(f"Chunk {chunk_num} completed successfully")
            
        except Exception as e:
            print(f"Error in chunk {chunk_num}: {str(e)}")
            raise
            
    return similarity_matrix

@tool(args_schema=StarkEvalInput)
def evaluate_stark_retrieval(
    query_file: str,
    node_file: str,
    batch_size: int = 256,
    split: str = "test-0.1"
) -> Dict[str, Any]:
    """Evaluate retrieval performance using StarkQA benchmark."""
    try:
        print("\n=== Starting Evaluation ===")
        
        # Load data
        print("\nStep 1: Loading data files")
        queries_df = pd.read_parquet(query_file)
        nodes_df = pd.read_parquet(node_file)
        print(f"✓ Loaded {len(queries_df)} queries and {len(nodes_df)} nodes")
        
        # Memory check
        query_size = sys.getsizeof(queries_df) / (1024 * 1024)
        node_size = sys.getsizeof(nodes_df) / (1024 * 1024)
        print(f"Memory usage - Queries: {query_size:.2f}MB, Nodes: {node_size:.2f}MB")
        
        # Get max node ID
        max_node_id = max(nodes_df.node_id)
        print(f"✓ Max node ID: {max_node_id}")
        
        # Process embeddings
        print("\nStep 2: Converting embeddings to tensors")
        try:
            query_embeddings = torch.tensor(
                np.stack([np.array(emb) for emb in queries_df.query_embedded.values]),
                dtype=torch.float32
            )
            node_embeddings = torch.tensor(
                np.stack([np.array(emb) for emb in nodes_df.x.values]),
                dtype=torch.float32
            )
            print(f"✓ Created tensors - Queries: {query_embeddings.shape}, Nodes: {node_embeddings.shape}")
        except Exception as e:
            print(f"Error creating tensors: {str(e)}")
            raise
        
        # Compute similarities
        print("\nStep 3: Computing similarities")
        try:
            similarity = compute_similarities_in_chunks(
                query_embeddings,
                node_embeddings,
                chunk_size=1000,
                device="cpu"
            )
            print("✓ Similarity computation complete")
        except Exception as e:
            print(f"Error computing similarities: {str(e)}")
            raise
        
        # Process node similarities
        print("\nStep 4: Computing node similarities")
        try:
            chunk_size = 500
            node_similarities = []
            total_chunks = (len(nodes_df) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(nodes_df), chunk_size):
                chunk_num = i // chunk_size + 1
                end_idx = min(i + chunk_size, len(nodes_df))
                print(f"Processing node chunk {chunk_num}/{total_chunks}")
                
                chunk_embeddings = node_embeddings[i:end_idx]
                chunk_similarities = torch.nn.functional.cosine_similarity(
                    chunk_embeddings.unsqueeze(1),
                    node_embeddings.unsqueeze(0),
                    dim=2
                )
                node_similarities.append(chunk_similarities.cpu())
            
            node_similarities = torch.cat(node_similarities, dim=0).numpy()
            print("✓ Node similarities computation complete")
        except Exception as e:
            print(f"Error computing node similarities: {str(e)}")
            raise
        
        # Compute metrics
        print("\nStep 5: Computing evaluation metrics")
        try:
            pred_ids = torch.LongTensor(nodes_df.node_id.tolist())
            pred = similarity.t()
            
            eval_results = []
            metrics = [
                "mrr", "map", "rprecision",
                "recall@5", "recall@10", "recall@20", "recall@50", "recall@100",
                "hit@1", "hit@3", "hit@5", "hit@10", "hit@20", "hit@50"
            ]
            
            total_batches = (len(queries_df) + batch_size - 1) // batch_size
            for batch_start in range(0, len(queries_df), batch_size):
                batch_num = batch_start // batch_size + 1
                batch_end = min(batch_start + batch_size, len(queries_df))
                print(f"Processing evaluation batch {batch_num}/{total_batches}")
                
                batch_queries = queries_df.iloc[batch_start:batch_end]
                answer_ids = [torch.LongTensor(aids) for aids in batch_queries.answer_ids]
                
                batch_results = evaluate_batch(
                    candidate_ids=pred_ids,
                    pred_ids=pred_ids,
                    pred=pred[:, batch_start:batch_end],
                    answer_ids=answer_ids,
                    metrics=metrics,
                    device="cpu"
                )
                
                for i, result in enumerate(batch_results):
                    result["query_id"] = batch_queries.iloc[i].id
                    eval_results.append(result)
            
            print("✓ Metrics computation complete")
        except Exception as e:
            print(f"Error computing metrics: {str(e)}")
            raise
        
        # Calculate final metrics
        print("\nStep 6: Computing final results")
        try:
            mean_metrics = {}
            for metric in metrics:
                mean_metrics[metric] = float(np.mean([r[metric] for r in eval_results]))
            
            shared_state.set(
                config.StateKeys.KNOWLEDGE_GRAPH,
                {
                    "nodes": nodes_df.to_dict("records"),
                    "similarities": node_similarities.tolist()
                }
            )
            shared_state.set(config.StateKeys.EVALUATION_RESULTS, eval_results)
            shared_state.set(config.StateKeys.METRICS, mean_metrics)
            
            print("✓ Final results computed and stored")
        except Exception as e:
            print(f"Error in final computations: {str(e)}")
            raise
        
        print("\n=== Evaluation Complete ===")
        return {
            "status": "success",
            "metrics": mean_metrics,
            "detailed_results": eval_results,
            "total_evaluated": len(eval_results),
            "nodes": len(nodes_df),
            "message": f"Successfully evaluated {len(eval_results)} queries"
        }
    
    except Exception as e:
        error_msg = f"Error in evaluation process: {str(e)}"
        print(f"\n!!! ERROR !!!\n{error_msg}")
        raise ToolException(error_msg)tion(f"Error in evaluation process: {str(e)}")


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
