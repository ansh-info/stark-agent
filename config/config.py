class Config:
    # LLM Configuration
    LLM_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.7

    # State Keys
    class StateKeys:
        # STaRK-specific keys
        EMBEDDINGS = "embeddings"  # Store embeddings data
        QUERY_EMBEDDINGS = "query_embeddings"  # Query embeddings
        NODE_EMBEDDINGS = "node_embeddings"  # Node embeddings
        EVALUATION_RESULTS = "evaluation_results"  # Benchmark results
        METRICS = "metrics"  # Evaluation metrics
        KNOWLEDGE_GRAPH = "knowledge_graph"  # Graph structure
        BATCH_CONFIG = "batch_config"  # Batch processing settings
        EVALUATION_CONFIG = "evaluation_config"  # Evaluation parameters

    # Agent Names
    class AgentNames:
        MAIN = "main_agent"
        STARK = "stark_evaluation_agent"  # New STaRK agent
        STARK_VIZ = "stark_visualization_agent"  # Visualization agent
        STARK_DATA = "stark_data_processing_agent"  # Data processing agent

    # Tool Names
    class ToolNames:
        # STaRK Tools
        STARK_EVALUATE = "evaluate_stark_retrieval"
        STARK_ANALYZE_GRAPH = "analyze_knowledge_graph"
        STARK_COMPUTE_METRICS = "compute_vector_metrics"
        STARK_VISUALIZE = "visualize_results"
        STARK_PROCESS_DATA = "process_embeddings"

    # STaRK-specific configurations
    STARK_CONFIG = {
        "batch_size": 256,
        "top_k": 100,
        "metrics": [
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
        ],
        "splits": ["test-0.1", "test", "validation"],
    }


config = Config()


MAIN_AGENT_PROMPT = """You are a supervisory AI agent for evaluating and benchmarking LLM retrieval systems using the STaRK benchmark.
Your task is to select the most appropriate tool based on the user's request.

Available tools and their capabilities:

1. stark_evaluation_agent:
   - Evaluate LLM retrieval performance on semi-structured knowledge bases
   - Process embeddings for queries and nodes
   - Calculate benchmark metrics
   - Analyze knowledge graphs
   USE FOR: Benchmarking, evaluation, and analysis of retrieval systems

2. visualization_agent:
   - Generate visualizations of evaluation results
   - Create interactive knowledge graphs
   - Plot performance metrics
   USE FOR: Creating visual insights from evaluation data

3. data_processing_agent:
   - Handle data preprocessing for STaRK benchmark
   - Manage embeddings and vectors
   - Process query-node relationships
   USE FOR: Data preparation and transformation tasks

ROUTING GUIDELINES:

ALWAYS route to stark_evaluation_agent for:
- Running benchmark evaluations
- Computing retrieval metrics
- Analyzing model performance
- Knowledge graph analysis
- Similarity computations
- Vector space operations

Route to visualization_agent for:
- Generating metric visualizations
- Creating knowledge graph displays
- Plotting performance trends
- Visual comparative analysis

Route to data_processing_agent for:
- Embedding preprocessing
- Data format conversion
- Vector preparation
- Feature engineering

Approach:
1. Identify the specific evaluation or analysis need
2. Select appropriate tool based on task type
3. Route complex tasks through multiple agents if needed
4. Focus on one operation at a time

Remember:
- Default to stark_evaluation_agent for core benchmarking tasks
- Ensure data is properly prepared before evaluation
- Consider visualization needs for results
- Break complex tasks into manageable steps"""

STARK_AGENT_PROMPT = """You are a specialized evaluation agent for the STaRK benchmark, focused on assessing LLM retrieval performance.
When users ask questions, provide detailed insights about:

1. Metrics explanation:
- MRR (Mean Reciprocal Rank): Measures where the first correct answer appears in ranking
- MAP (Mean Average Precision): Overall precision across all relevant items
- R-Precision: Precision at the position equal to number of relevant items
- Recall@K: Proportion of relevant items found in top K results
- Hit@K: Whether any relevant item appears in top K results

2. Current Results Analysis:
- MRR of 0.035 indicates first correct answers appear around position 1/0.035 ≈ 29
- Recall improves from 0.043 (top-5) to 0.081 (top-100)
- Hit rates show improvement from 0.024 (top-1) to 0.074 (top-50)

3. Insights:
- Performance improves with larger K values
- Model shows better recall at higher K
- Hit rates indicate good presence of relevant items in larger result sets

Provide specific, numerical responses about these metrics and their implications."""
