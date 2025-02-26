import os
import sys
from typing import Any, Dict, List

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from agents.main_agent import main_agent
from config.config import config
from state.shared_state import shared_state
from tools.stark.evaluation_retrival import (
    evaluate_stark_retrieval,
)  # Note the correct filename


def render_metrics_visualization(metrics: dict):
    """Render evaluation metrics visualization"""
    # Create bar chart for metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    fig = px.bar(metrics_df, x="Metric", y="Value", title="Evaluation Metrics")
    st.plotly_chart(fig)

    # Create detailed metrics table
    st.dataframe(metrics_df)


def render_knowledge_graph(nodes_df):
    """Render knowledge graph visualization"""
    import networkx as nx

    # Get similarities from shared state
    graph_data = shared_state.get(config.StateKeys.KNOWLEDGE_GRAPH)
    if not graph_data:
        st.warning("No knowledge graph data available")
        return

    # Create graph
    G = nx.Graph()

    # Add nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], name=row["node_name"], type=row["node_type"])

    # Add edges based on similarity
    similarity_matrix = np.array(graph_data["similarities"])
    threshold = 0.7  # High similarity threshold
    num_nodes = len(nodes_df)

    # Add top K most similar edges for each node
    K = 5  # Number of edges per node
    for i in range(num_nodes):
        # Get top K similar nodes
        similarities = similarity_matrix[i]
        top_k_indices = np.argsort(similarities)[-K - 1 :]  # Get K+1 for excluding self

        for j in top_k_indices:
            if i < j and similarities[j] > threshold:  # Avoid duplicate edges
                G.add_edge(
                    nodes_df.iloc[i]["node_id"],
                    nodes_df.iloc[j]["node_id"],
                    weight=float(similarities[j]),
                )

    # Create layout
    pos = nx.spring_layout(G)

    # Create plotly figure
    edge_x = []
    edge_y = []
    edge_weights = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.extend([edge[2]["weight"], edge[2]["weight"], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_info = nodes_df[nodes_df.node_id == node].iloc[0]
        node_text.append(
            f"ID: {node}<br>Name: {node_info['node_name']}<br>Type: {node_info['node_type']}"
        )
        node_colors.append(hash(node_info["node_type"]) % 20)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=10,
            color=node_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Node Types"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Knowledge Graph Visualization (showing top {K} connections per node)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Note: Edges show strong similarity connections",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig)


def process_chat_query(query: str) -> str:
    """Process chat queries through the agent hierarchy"""
    try:
        # Route through main agent
        response = main_agent.invoke({"messages": [{"role": "user", "content": query}]})

        # Check if we have evaluation results
        eval_results = shared_state.get(config.StateKeys.EVALUATION_RESULTS)
        if eval_results and "How can I help you with the evaluation?" in response.get(
            "response", ""
        ):
            metrics = shared_state.get(config.StateKeys.METRICS)
            # Provide context-aware response
            return f"""Based on our evaluation results:
- MRR: {metrics['mrr']:.3f} (ranks first correct answer around position {1/metrics['mrr']:.0f})
- MAP: {metrics['map']:.3f}
- Recall@100: {metrics['recall@100']:.3f} (finding {metrics['recall@100']*100:.1f}% of relevant items)

How can I help you understand these metrics better?"""

        return response.get(
            "response", "I'm here to help you understand the evaluation results."
        )
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"


def render_evaluation_section():
    """Render the evaluation section of the UI"""
    st.header("Evaluation")

    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            try:
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f1:
                    f1.write(query_file.getvalue())
                    query_path = f1.name

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2:
                    f2.write(node_file.getvalue())
                    node_path = f2.name

                # Run evaluation
                result = evaluate_stark_retrieval.invoke(
                    {
                        "query_file": query_path,
                        "node_file": node_path,
                        "batch_size": batch_size,
                        "split": split,
                    }
                )

                if result["status"] == "success":
                    st.success("✅ Evaluation completed successfully!")

                    with st.expander("Evaluation Metrics", expanded=True):
                        # Display metrics
                        render_metrics_visualization(result["metrics"])

                    with st.expander("Knowledge Graph", expanded=True):
                        # Get graph data from shared state
                        graph_data = shared_state.get(config.StateKeys.KNOWLEDGE_GRAPH)
                        if graph_data:
                            nodes_df = pd.DataFrame(graph_data["nodes"])
                            render_knowledge_graph(nodes_df, graph_data["similarities"])
                        else:
                            st.warning("No knowledge graph data available")

                else:
                    st.error(
                        f"Evaluation failed: {result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")


def main():
    st.title("STaRK: Benchmarking LLM Retrieval")

    # Create status container
    status_container = st.container()
    progress_bar = status_container.progress(0)
    status_text = status_container.empty()
    metrics_container = st.container()

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        batch_size = st.slider("Batch Size", 32, 512, 256, 32)
        split = st.selectbox(
            "Evaluation Split", options=["test-0.1", "test", "validation"]
        )

        # Add memory info
        st.info(
            """
        File Size Limits:
        - Query Embeddings: ~12MB
        - Node Embeddings: ~200MB
        Processing larger files may take several minutes.
        """
        )

    # File upload section
    st.header("Data Upload")
    col1, col2 = st.columns(2)

    with col1:
        query_file = st.file_uploader(
            "Upload Query Embeddings (Parquet)",
            type=["parquet"],
            help="Upload query embeddings file (expects around 12MB)",
        )
        if query_file:
            size_mb = query_file.size / (1024 * 1024)
            st.success(f"Query file loaded: {size_mb:.2f}MB")

    with col2:
        node_file = st.file_uploader(
            "Upload Node Embeddings (Parquet)",
            type=["parquet"],
            help="Upload node embeddings file (expects around 200MB)",
        )
        if node_file:
            size_mb = node_file.size / (1024 * 1024)
            st.success(f"Node file loaded: {size_mb:.2f}MB")

    # Evaluation section
    if query_file and node_file:
        st.header("Evaluation")

        if st.button("Run Evaluation"):
            try:
                # Save uploaded files
                status_text.text("Step 1/5: Saving uploaded files...")
                progress_bar.progress(10)

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f1:
                    f1.write(query_file.getvalue())
                    query_path = f1.name

                progress_bar.progress(20)
                status_text.text("Step 2/5: Processing query embeddings...")

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2:
                    f2.write(node_file.getvalue())
                    node_path = f2.name

                progress_bar.progress(30)
                status_text.text("Step 3/5: Processing node embeddings...")

                # Run evaluation with progress updates
                status_text.text(
                    "Step 4/5: Running evaluation (this may take several minutes)..."
                )
                progress_bar.progress(40)

                result = evaluate_stark_retrieval.invoke(
                    {
                        "query_file": query_path,
                        "node_file": node_path,
                        "batch_size": batch_size,
                        "split": split,
                    }
                )

                if result["status"] == "success":
                    progress_bar.progress(90)
                    status_text.text("Step 5/5: Generating visualizations...")

                    with metrics_container:
                        st.success("✅ Evaluation completed successfully!")

                        # Display metrics
                        st.subheader("Evaluation Metrics")

                        # Display summary metrics first
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MRR", f"{result['metrics']['mrr']:.3f}")
                        with col2:
                            st.metric("MAP", f"{result['metrics']['map']:.3f}")
                        with col3:
                            st.metric(
                                "R-Precision", f"{result['metrics']['rprecision']:.3f}"
                            )

                        # Display detailed metrics visualization
                        render_metrics_visualization(result["metrics"])

                        # Load and display knowledge graph
                        st.subheader("Knowledge Graph Visualization")
                        nodes_df = pd.read_parquet(node_path)
                        render_knowledge_graph(nodes_df)

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")

                else:
                    st.error(
                        f"Evaluation failed: {result.get('message', 'Unknown error')}"
                    )

            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")

            finally:
                # Cleanup temporary files
                try:
                    os.unlink(query_path)
                    os.unlink(node_path)
                except:
                    pass

    # Add processing tips
    st.sidebar.markdown(
        """
    ### Processing Time:
    - Small files (~10MB): 1-2 minutes
    - Medium files (~100MB): 3-5 minutes
    - Large files (>200MB): 5+ minutes
    
    ### Tips:
    - Use smaller batch sizes for large files
    - Keep the browser tab active
    - Check terminal for detailed logs
    """
    )

    # Chat interface
    st.header("Chat Interface")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the evaluation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                response = main_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                content = response.get(
                    "response", "How can I help you with the evaluation?"
                )
                st.markdown(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )
            except Exception as e:
                st.error(
                    "Sorry, I couldn't process that request. Please try asking about the evaluation process or results."
                )


if __name__ == "__main__":
    main()
