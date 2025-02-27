from typing import Any, Dict


class Config:
    # LLM Configuration
    LLM_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.7

    # State Keys
    class StateKeys:
        # Existing keys
        CURRENT_TOOL = "current_tool"
        CURRENT_AGENT = "current_agent"
        RESPONSE = "response"
        ERROR = "error"
        CHAT_HISTORY = "chat_history"
        USER_INFO = "user_info"
        MEMORY = "memory"

        # StarkQA specific keys
        EMBEDDINGS = "embeddings"
        QUERY_EMBEDDINGS = "query_embeddings"
        NODE_EMBEDDINGS = "node_embeddings"
        EVALUATION_RESULTS = "evaluation_results"
        METRICS = "metrics"
        KNOWLEDGE_GRAPH = "knowledge_graph"
        BATCH_CONFIG = "batch_config"
        EVALUATION_CONFIG = "evaluation_config"

    # Agent Names
    class AgentNames:
        MAIN = "main_agent"
        STARK = "stark_evaluation_agent"
        STARK_VIZ = "stark_visualization_agent"
        STARK_DATA = "stark_data_processing_agent"

    # Tool Names
    class ToolNames:
        # StarkQA Tools
        STARK_EVALUATE = "evaluate_stark_retrieval"
        STARK_ANALYZE_GRAPH = "analyze_knowledge_graph"
        STARK_COMPUTE_METRICS = "compute_vector_metrics"
        STARK_VISUALIZE = "visualize_results"
        STARK_PROCESS_DATA = "process_embeddings"

    # Main Agent Prompt
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

    # StarkQA Agent Prompt
    STARK_AGENT_PROMPT = """You are a specialized evaluation agent for the STaRK benchmark, focused on assessing LLM retrieval performance.
You have access to these tools:

1. evaluate_stark_retrieval:
   USE FOR: Core benchmark evaluation
   - Process query and node embeddings
   - Calculate similarity scores
   - Compute standard metrics (MRR, MAP, NDCG)
   - Generate performance reports

2. analyze_knowledge_graph:
   USE FOR: Graph structure analysis
   - Examine node relationships
   - Calculate graph metrics
   - Identify important patterns
   - Assess connectivity

3. compute_vector_metrics:
   USE FOR: Vector space analysis
   - Calculate embedding similarities
   - Analyze vector distributions
   - Measure semantic relationships
   - Evaluate vector quality

GUIDELINES:

For benchmark evaluation:
- Verify input data formats
- Apply appropriate metrics
- Consider all evaluation aspects
- Generate comprehensive reports

For knowledge graph analysis:
- Focus on relevant subgraphs
- Identify key relationships
- Analyze node importance
- Consider graph properties

For vector analysis:
- Check embedding quality
- Consider similarity measures
- Evaluate vector spaces
- Assess semantic relevance

Best practices:
1. Validate input data quality
2. Use appropriate evaluation metrics
3. Consider multiple performance aspects
4. Generate clear insights
5. Enable result visualization

Remember:
- Focus on benchmark objectives
- Ensure metric accuracy
- Consider practical implications
- Provide actionable insights"""

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
