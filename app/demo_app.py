import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


# Import your existing components
from agents.main_agent import main_agent
from config.config import config
from state.shared_state import shared_state
from tools.stark.evaluation_retrival import (
    evaluate_stark_retrieval,
    evaluate_stark_retrieval_remote,
)

# Available embedding models
EMBEDDING_MODELS = {
    "OpenAI Ada 002": "text-embedding-ada-002",
    "Cohere Embed": "cohere-embed-multilingual-v3.0",
    "Anthropic Claude": "claude-3-embedding",
    "Custom Model": "custom",
}

# Available LLM models
LLM_MODELS = {
    "GPT-4": "gpt-4",
    "Claude-3 Opus": "claude-3-opus",
    "Claude-3 Sonnet": "claude-3-sonnet",
    "Mistral Large": "mistral-large",
    "Custom Model": "custom",
}


def set_page_config():
    """Configure the Streamlit page settings"""
    st.set_page_config(
        page_title="STaRK Benchmark Suite",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_welcome_page():
    """Render the welcome/landing page"""
    st.title("🎯 STaRK Benchmark Suite")

    st.markdown(
        """
    ### Transform Your Knowledge Base Evaluation with STaRK
    
    STaRK (Semi-structured Text and Relational Knowledge) is a comprehensive suite for 
    evaluating and enhancing LLM retrieval systems on semi-structured knowledge bases.
    
    #### 🎯 Who is this for?
    - **ML Engineers** evaluating retrieval system performance
    - **Data Scientists** analyzing knowledge graph relationships
    - **Researchers** benchmarking LLM capabilities
    - **Product Teams** optimizing search and retrieval
    
    #### 🚀 Key Features
    1. **Data Enrichment** - Enhance your datasets with advanced features
    2. **Evaluation Pipeline** - Comprehensive retrieval metrics
    3. **Visual Analytics** - Interactive knowledge graph visualization
    4. **Query Interface** - Natural language queries against your knowledge base
    
    #### 💡 Getting Started
    1. Navigate to the desired tool using the sidebar
    2. Upload your data or use our sample datasets
    3. Configure your evaluation parameters
    4. Analyze results through interactive visualizations
    """
    )

    # Display feature boxes
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "📊 **Evaluation Metrics**\n\n"
            "- MRR (Mean Reciprocal Rank)\n"
            "- MAP (Mean Average Precision)\n"
            "- NDCG (Normalized DCG)\n"
            "- Recall@K\n"
            "- Precision@K"
        )

    with col2:
        st.success(
            "🔍 **Analysis Tools**\n\n"
            "- Knowledge Graph Analysis\n"
            "- Similarity Computing\n"
            "- Subgraph Extraction\n"
            "- Pattern Recognition"
        )

    with col3:
        st.warning(
            "🛠️ **Customization**\n\n"
            "- Custom Embeddings\n"
            "- Model Selection\n"
            "- Batch Processing\n"
            "- Export Options"
        )


def render_data_enrichment():
    """Render the data enrichment page"""
    st.title("📈 Data Enrichment")

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Your Dataset", type=["csv", "parquet", "json"]
    )

    if uploaded_file:
        # Load data
        if uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)

        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Enrichment options
        st.subheader("Enrichment Options")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Feature Generation")
            generate_embeddings = st.checkbox("Generate Embeddings")
            compute_statistics = st.checkbox("Compute Statistics")
            extract_keywords = st.checkbox("Extract Keywords")

        with col2:
            st.markdown("#### Model Selection")
            if generate_embeddings:
                embedding_model = st.selectbox(
                    "Select Embedding Model", list(EMBEDDING_MODELS.keys())
                )

        if st.button("Enrich Dataset"):
            # Add your enrichment logic here
            st.success("Dataset enriched successfully!")


def render_evaluation():
    """Render the evaluation page"""
    st.title("🎯 Evaluation Pipeline")

    # Configuration section
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            batch_size = st.slider(
                "Batch Size",
                32,
                512,
                256,
                32,
                help="Number of samples processed at once",
            )
            processing_type = st.radio(
                "Processing Type",
                ["Remote (GPT-4o-mini)", "Local"],
                help="Choose where to run the evaluation",
            )

        with col2:
            split = st.selectbox(
                "Evaluation Split",
                options=["test-0.1", "test", "validation"],
                help="Dataset split to use for evaluation",
            )
            model = st.selectbox(
                "LLM Model", list(LLM_MODELS.keys()), help="Model to use for evaluation"
            )

    # File upload section
    col1, col2 = st.columns(2)

    with col1:
        query_file = st.file_uploader(
            "Upload Query Embeddings (Parquet)",
            type=["parquet"],
            help="Upload query embeddings file",
        )
        if query_file:
            query_size = len(query_file.getvalue()) / (1024 * 1024)
            st.success(f"Query file loaded: {query_size:.2f}MB")

    with col2:
        node_file = st.file_uploader(
            "Upload Node Embeddings (Parquet)",
            type=["parquet"],
            help="Upload node embeddings file",
        )
        if node_file:
            node_size = len(node_file.getvalue()) / (1024 * 1024)
            st.success(f"Node file loaded: {node_size:.2f}MB")

    if query_file and node_file:
        if st.button("Run Evaluation"):
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()

            try:
                with st.spinner("Processing evaluation..."):
                    if processing_type == "Remote (GPT-4o-mini)":
                        # Convert files to base64
                        query_content = base64.b64encode(query_file.getvalue()).decode()
                        node_content = base64.b64encode(node_file.getvalue()).decode()

                        # Call remote evaluation
                        result = evaluate_stark_retrieval_remote(
                            query_file_content=query_content,
                            node_file_content=node_content,
                            batch_size=batch_size,
                            split=split,
                            model=LLM_MODELS[model],
                        )
                    else:
                        # Use local evaluation
                        with tempfile.NamedTemporaryFile(
                            suffix=".parquet", delete=False
                        ) as f1:
                            f1.write(query_file.getvalue())
                            query_path = f1.name

                        with tempfile.NamedTemporaryFile(
                            suffix=".parquet", delete=False
                        ) as f2:
                            f2.write(node_file.getvalue())
                            node_path = f2.name

                        result = evaluate_stark_retrieval(
                            query_file=query_path,
                            node_file=node_path,
                            batch_size=batch_size,
                            split=split,
                        )

                    if result["status"] == "success":
                        st.success("✅ Evaluation completed successfully!")

                        # Store results in shared state
                        shared_state.set(
                            config.StateKeys.EVALUATION_RESULTS,
                            result.get("detailed_results"),
                        )
                        shared_state.set(
                            config.StateKeys.METRICS, result.get("metrics")
                        )

                        # Display metrics
                        st.subheader("Evaluation Metrics")
                        cols = st.columns(4)
                        for i, (metric, value) in enumerate(result["metrics"].items()):
                            with cols[i % 4]:
                                st.metric(label=metric.upper(), value=f"{value:.3f}")

                        # Add link to full report
                        st.markdown("[View Full Report](#visualization)")
                    else:
                        st.error(
                            f"Evaluation failed: {result.get('message', 'Unknown error')}"
                        )

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

            finally:
                # Cleanup temporary files if using local processing
                if processing_type == "Local":
                    try:
                        os.unlink(query_path)
                        os.unlink(node_path)
                    except:
                        pass


def render_visualization():
    """Render the visualization page"""
    st.title("📊 Results & Visualization")

    # Check if we have evaluation results
    eval_results = shared_state.get(config.StateKeys.EVALUATION_RESULTS)
    metrics = shared_state.get(config.StateKeys.METRICS)

    if not eval_results:
        st.warning("No evaluation results available. Please run an evaluation first.")
        return

    # Display metrics
    st.subheader("📊 Evaluation Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("MRR", f"{metrics['mrr']:.3f}")
    with col2:
        st.metric("MAP", f"{metrics['map']:.3f}")
    with col3:
        st.metric("R-Precision", f"{metrics['rprecision']:.3f}")

    # Detailed visualizations
    st.subheader("📈 Detailed Analysis")

    # Add your existing visualization code here


def render_query_interface():
    """Render the query interface page"""
    st.title("🔍 Query Interface")

    # Model selection
    model = st.selectbox("Select Model", list(LLM_MODELS.keys()))

    # Query input
    query = st.text_area("Enter your query")

    if st.button("Execute Query"):
        if not query:
            st.warning("Please enter a query")
            return

        # Add your query processing logic here
        st.success("Query executed successfully!")


def render_demo_page():
    """Render the demo page with sample visualizations and reports"""
    st.title("🎮 Interactive Demo")

    # Sample metrics
    st.header("📊 Sample Benchmark Results")

    # Create sample metrics
    sample_metrics = {
        "mrr": 0.847,
        "map": 0.762,
        "rprecision": 0.834,
        "recall@5": 0.891,
        "recall@10": 0.923,
        "hit@1": 0.756,
        "hit@3": 0.845,
        "hit@5": 0.912,
    }

    # Display metrics in columns
    cols = st.columns(4)
    for i, (metric, value) in enumerate(sample_metrics.items()):
        with cols[i % 4]:
            st.metric(
                label=metric.upper(),
                value=f"{value:.3f}",
                delta=f"{(value - 0.7):.3%}" if value > 0.7 else f"{(value - 0.7):.3%}",
            )

    # Sample Knowledge Graph
    st.header("🕸️ Sample Knowledge Graph")

    # Create sample graph data
    num_nodes = 50
    np.random.seed(42)

    # Generate sample node positions
    node_pos = {i: [np.random.rand(), np.random.rand()] for i in range(num_nodes)}

    # Generate sample edges
    edges = []
    edge_weights = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 5, num_nodes)):
            if np.random.rand() > 0.5:
                edges.append((i, j))
                edge_weights.append(np.random.rand())

    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = node_pos[edge[0]]
        x1, y1 = node_pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_x = []
    node_y = []
    for pos in node_pos.values():
        node_x.append(pos[0])
        node_y.append(pos[1])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="YlGnBu",
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
        ),
    )

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Sample Report
    st.header("📑 Sample Evaluation Report")

    with st.expander("Performance Analysis", expanded=True):
        st.markdown(
            """
        ### STaRK Benchmark Evaluation Report
        
        #### Overview
        The evaluation was performed on a knowledge base containing:
        - 10,734 nodes
        - 11,204 queries
        - 384-dimensional embeddings
        
        #### Key Findings
        1. **Retrieval Performance**
           - Achieved MRR of 0.847, indicating strong ranking performance
           - MAP score of 0.762 shows good overall precision
           - High recall@k metrics suggest effective coverage
        
        2. **Knowledge Graph Analysis**
           - Average node degree: 4.2
           - Graph density: 0.083
           - Clustering coefficient: 0.412
        
        3. **Query Performance**
           - Average response time: 142ms
           - 95th percentile latency: 267ms
           - Query throughput: 70 QPS
        
        #### Recommendations
        1. Increase embedding dimensions for better representation
        2. Optimize index for faster retrieval
        3. Add more domain-specific training data
        """
        )

    # Interactive query demo
    st.header("🔍 Sample Queries")

    # Sample queries and their results
    sample_queries = [
        "Find all nodes related to EZH2 Gene",
        "Show subgraph of epithelioid sarcoma",
        "List most connected nodes",
    ]

    selected_query = st.selectbox("Try a sample query:", sample_queries)

    if st.button("Run Sample Query"):
        st.success(f"Query executed: {selected_query}")

        # Show sample results
        if selected_query == sample_queries[0]:
            st.json(
                {
                    "related_nodes": [
                        {"id": 1, "name": "4-HDJAHDJHKAD-6-FDAD", "relevance": 0.92},
                        {"id": 5, "name": "Paclitaxel", "relevance": 0.89},
                        {"id": 12, "name": "Heart disease", "relevance": 0.78},
                    ]
                }
            )
        elif selected_query == sample_queries[1]:
            st.image(
                "images/epithelioid_sarcoma.jpg",  # Replace with the correct local path
                caption="Epithelioid Sarcoma - Very High Magnification",
            )
        else:
            st.table(
                pd.DataFrame(
                    {
                        "Node": ["Node A", "Node B", "Node C"],
                        "Connections": [15, 12, 10],
                        "Centrality": [0.92, 0.85, 0.79],
                    }
                )
            )


def main():
    """Main application function"""
    set_page_config()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Welcome",
            "Demo",
            "Data Enrichment",
            "Evaluation",
            "Visualization",
            "Query Interface",
        ],
    )

    # Page routing
    if page == "Welcome":
        render_welcome_page()
    elif page == "Demo":
        render_demo_page()
    elif page == "Data Enrichment":
        render_data_enrichment()
    elif page == "Evaluation":
        render_evaluation()
    elif page == "Visualization":
        render_visualization()
    elif page == "Query Interface":
        render_query_interface()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Made with ❤️ by Your Team")


if __name__ == "__main__":
    main()
