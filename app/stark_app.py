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


def render_knowledge_graph(nodes_df, similarity_matrix=None):
    """Render knowledge graph visualization"""
    import networkx as nx

    # Create graph
    G = nx.Graph()

    # Add nodes
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], name=row["node_name"], type=row["node_type"])

    # Add edges based on similarity if available
    if similarity_matrix is not None:
        threshold = 0.5  # Similarity threshold
        edges = []
        for i in range(len(nodes_df)):
            for j in range(i + 1, len(nodes_df)):
                if similarity_matrix[i, j] > threshold:
                    edges.append(
                        (
                            nodes_df.iloc[i]["node_id"],
                            nodes_df.iloc[j]["node_id"],
                            similarity_matrix[i, j],
                        )
                    )
        G.add_weighted_edges_from(edges)

    # Create layout
    pos = nx.spring_layout(G)

    # Create plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            ),
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                text=[f"{nodes_df.iloc[i]['node_name']}" for i in range(len(nodes_df))],
                marker=dict(
                    size=10,
                    color=nodes_df["node_type"].astype("category").cat.codes,
                    colorscale="Viridis",
                    showscale=True,
                ),
            ),
        ],
        layout=go.Layout(
            title="Knowledge Graph",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    st.plotly_chart(fig)


def main():
    st.title("STaRK: Benchmarking LLM Retrieval")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    batch_size = st.sidebar.slider("Batch Size", 32, 512, 256, 32)
    split = st.sidebar.selectbox(
        "Evaluation Split", options=["test-0.1", "test", "validation"]
    )

    # File upload section
    st.header("Data Upload")
    col1, col2 = st.columns(2)

    with col1:
        query_file = st.file_uploader(
            "Upload Query Embeddings (Parquet)", type=["parquet"]
        )

    with col2:
        node_file = st.file_uploader(
            "Upload Node Embeddings (Parquet)", type=["parquet"]
        )

    # Evaluation section
    if query_file and node_file:
        st.header("Evaluation")

        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                # Save uploaded files temporarily
                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f1:
                    f1.write(query_file.getvalue())
                    query_path = f1.name

                with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f2:
                    f2.write(node_file.getvalue())
                    node_path = f2.name

                try:
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
                        st.success("Evaluation completed successfully!")

                        # Display metrics
                        st.subheader("Evaluation Metrics")
                        render_metrics_visualization(result["metrics"])

                        # Load node data for visualization
                        nodes_df = pd.read_parquet(node_path)

                        # Display knowledge graph
                        st.subheader("Knowledge Graph")
                        render_knowledge_graph(nodes_df)

                    else:
                        st.error(
                            f"Evaluation failed: {result.get('message', 'Unknown error')}"
                        )

                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")

                finally:
                    # Cleanup temporary files
                    import os

                    os.unlink(query_path)
                    os.unlink(node_path)

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
            response = main_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            content = response.get(
                "response", "Sorry, I couldn't process that request."
            )
            st.markdown(content)
            st.session_state.messages.append({"role": "assistant", "content": content})


if __name__ == "__main__":
    main()
