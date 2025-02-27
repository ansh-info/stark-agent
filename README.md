# STaRK Benchmark Evaluation

<p align="center">
  <img src="https://stark.stanford.edu/images/logo.png" alt="STaRK Logo" width="200"/>
</p>

<p align="center">
  <strong>A hierarchical agent system for evaluating LLM retrieval performance on semi-structured knowledge bases.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#usage">Usage</a> •
  <a href="#evaluation-metrics">Metrics</a> •
  <a href="#why-stark">Why STaRK?</a>
</p>

## 📋 Overview

STaRK (Semi-structured Text and Relational Knowledge) is a comprehensive benchmark designed to evaluate how well large language models (LLMs) and retrieval systems work with semi-structured knowledge bases (SKBs). These knowledge bases combine structured data (e.g., entity relationships) with unstructured data (e.g., textual descriptions), representing real-world knowledge complexity.

This project implements a hierarchical LLM-powered agent system to process, evaluate, and visualize the performance of retrieval systems on the STaRK benchmark, spanning three key domains:

1. **Product Search**: Detailed product metadata, reviews, and relationships
2. **Academic Paper Search**: Paper citations, author relationships, and content
3. **Precision Medicine**: Drug-disease interactions and clinical trial data

## 🌟 Features

- **Hierarchical Agent System**: Multi-agent architecture with specialized agents for different tasks
- **Comprehensive Evaluation**: Calculate standard retrieval metrics (MRR, MAP, NDCG, Recall@K, etc.)
- **Interactive Knowledge Graph**: Visualize node relationships and similarities
- **Streamlit UI**: User-friendly interface for uploading data, running evaluations, and analyzing results
- **Memory Optimization**: Efficiently process large embedding datasets with chunked computation
- **LLM-powered Analysis**: Natural language interface to query and understand results
- **Data Enrichment**: Tools to enhance datasets with additional features
- **Remote Processing**: Support for both local and remote (GPT-4o-mini) evaluation

## 🧠 Architecture

The system implements a hierarchical agent structure:

### Data Flow Diagram with Remote Processing

```mermaid
graph TD
    A[User] -->|Upload Embeddings| B[Streamlit UI]
    B -->|Configure Settings| B1[Local/Remote Selection]
    B -->|Trigger Evaluation| C[Main Agent]

    C -->|Route Request| D[StarkQA Agent]

    D -->|Local Processing| E1[Local Evaluation Tool]
    D -->|Remote Processing| E2[GPT-4o-mini API]

    E1 -->|Read| F1[Query Embeddings]
    E1 -->|Read| F2[Node Embeddings]
    E1 -->|Process| G1[Local Similarity Computation]

    E2 -->|Send| F3[Base64 Encoded Files]
    E2 -->|Process| G2[Remote Computation]

    G1 -->|Calculate| H1[Local Metrics]
    G2 -->|Calculate| H2[Remote Metrics]

    G1 -->|Generate| I1[Knowledge Graph]

    H1 -->|Store In| J[Shared State]
    H2 -->|Store In| J
    I1 -->|Store In| J

    J -->|Display In| K1[Metrics Visualization]
    J -->|Display In| K2[Graph Visualization]
    J -->|Reference For| K3[Chat Interface]

    K1 -->|Show In| B
    K2 -->|Show In| B
    K3 -->|Show In| B

    %% Styling
    classDef user fill:#f96,stroke:#333,stroke-width:2px
    classDef ui fill:#9cf,stroke:#333,stroke-width:2px
    classDef agent fill:#fcf,stroke:#333,stroke-width:2px
    classDef local fill:#cfc,stroke:#333,stroke-width:2px
    classDef remote fill:#f99,stroke:#333,stroke-width:2px
    classDef data fill:#fc9,stroke:#333,stroke-width:2px
    classDef process fill:#ff9,stroke:#333,stroke-width:2px
    classDef state fill:#c9f,stroke:#333,stroke-width:2px
    classDef output fill:#9c9,stroke:#333,stroke-width:2px

    class A user
    class B,B1 ui
    class C,D agent
    class E1,G1,H1,I1 local
    class E2,G2,H2 remote
    class F1,F2,F3 data
    class J state
    class K1,K2,K3 output
```

### Agents

- **Main Agent**: Supervisory agent that routes user queries and orchestrates tasks
- **StarkQA Agent**: Specialized agent focused on evaluation and metrics
- **Query Agent**: Handles natural language queries against the knowledge graph
- **Enrichment Agent**: Provides data enhancement capabilities

### Tools

- **Evaluation Tool**: Processes embeddings and computes similarities
- **Metrics Tool**: Calculates standard retrieval metrics
- **Visualization Tool**: Generates interactive visualizations
- **Knowledge Graph Tool**: Builds and queries the knowledge graph
- **Feature Generation Tool**: Enhances datasets with additional features

## 📊 Evaluation Pipeline

1. **Data Loading**: Process query and node embeddings from parquet files
2. **Similarity Computation**: Calculate similarities between queries and nodes using optimized batching
3. **Metrics Calculation**: Compute standard retrieval metrics (MRR, MAP, Recall@K)
4. **Knowledge Graph Generation**: Create interactive visualizations of node relationships
5. **Result Analysis**: Provide natural language insights about evaluation results

## Memory-Optimized Component Architecture

```mermaid
graph TD
    subgraph "UI Layer"
        A1[File Upload]
        A2[Configuration Settings]
        A3[Results Display]
        A4[Chat Interface]
        A5[Demo & Samples]
    end

    subgraph "Agent Layer"
        B1[Main Agent]
        B2[StarkQA Agent]
        B3[Enrichment Agent]
        B4[Query Agent]
    end

    subgraph "Tool Layer"
        C1[Evaluation Tool]
        C2[Knowledge Graph Generator]
        C3[Metrics Calculator]
        C4[Feature Generator]
        C5[Query Processor]
    end

    subgraph "Computation Layer"
        D1[Local Processing]
        D2[Remote GPT-4o-mini]
        D3[Memory Optimization]
    end

    subgraph "Data Layer"
        E1[Query Embeddings]
        E2[Node Embeddings]
        E3[Shared State]
        E4[Visualization Cache]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2
    B1 --> B3
    B1 --> B4

    B2 --> C1
    B2 --> C2
    B2 --> C3
    B3 --> C4
    B4 --> C5

    C1 --> D1
    C1 --> D2
    C2 --> D1
    C3 --> D1
    C3 --> D2
    C4 --> D1
    C5 --> D1

    D1 --> D3
    D3 --> E1
    D3 --> E2
    D1 --> E3
    D2 --> E3

    E3 --> E4
    E4 --> A3
    E4 --> A4
    E3 --> A5

    %% Styling
    classDef ui fill:#9cf,stroke:#333,stroke-width:2px
    classDef agent fill:#fcf,stroke:#333,stroke-width:2px
    classDef tool fill:#ff9,stroke:#333,stroke-width:2px
    classDef compute fill:#f99,stroke:#333,stroke-width:2px
    classDef data fill:#cfc,stroke:#333,stroke-width:2px

    class A1,A2,A3,A4,A5 ui
    class B1,B2,B3,B4 agent
    class C1,C2,C3,C4,C5 tool
    class D1,D2,D3 compute
    class E1,E2,E3,E4 data
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- LangChain/LangGraph
- OpenAI API key (for GPT-4-mini)
- PyTorch
- Streamlit

### Installation

```bash
# Clone the repository
git clone https://github.com/ansh-info/stark-agent
cd stark-agent

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your OpenAI API key
export OPENAI_API_KEY=your_api_key_here  # On Windows: set OPENAI_API_KEY=your_api_key_here
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app/demo_app.py
```

## 📝 Usage

1. **Upload Files**:

   - Query Embeddings (parquet format)
   - Node Embeddings (parquet format)

2. **Configure Settings**:

   - Batch Size: Number of queries to process at once
   - Processing Type: Local or Remote (GPT-4o-mini)
   - Evaluation Split: Data split to evaluate on
   - Model Selection: Choose LLM for analysis

3. **Run Evaluation**:

   - Click "Run Evaluation" to start the process
   - Monitor progress in the UI

4. **Analyze Results**:
   - View metrics visualization
   - Explore the knowledge graph
   - Ask questions in natural language

## 📈 Evaluation Metrics

The system calculates and displays the following metrics:

| Metric          | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| **MRR**         | Mean Reciprocal Rank - Measures where first correct answers appear in ranking |
| **MAP**         | Mean Average Precision - Measures precision across all relevant items         |
| **R-Precision** | Precision at the position equal to number of relevant items                   |
| **Recall@K**    | Proportion of relevant items found in top K results                           |
| **Hit@K**       | Whether any relevant item appears in top K results                            |
| **NDCG**        | Normalized Discounted Cumulative Gain - Evaluates ranking quality             |

## Evaluation Process

```mermaid
graph TD
    A[Start Evaluation] --> B1{Local or Remote?}

    B1 -->|Local| C1[Load Parquet Files]
    B1 -->|Remote| C2[Encode Files to Base64]

    C1 --> D1[Parse Answer IDs]
    C2 --> D2[Send to GPT-4o-mini]

    D1 --> E1[Convert to Tensors]
    D2 --> E2[Await Remote Response]

    E1 --> F1[Compute Similarities in Chunks]
    F1 --> G1[Calculate Memory-Optimized Metrics]

    E2 --> G2[Process Remote Results]

    G1 --> H1[Store Local Results]
    G2 --> H2[Store Remote Results]

    H1 --> I[Generate Visualizations]
    H2 --> I

    I --> J1[Interactive Knowledge Graph]
    I --> J2[Metric Dashboards]
    I --> J3[Performance Reports]

    J1 --> K[End Evaluation]
    J2 --> K
    J3 --> K

    %% Styling
    classDef start fill:#9f9,stroke:#333,stroke-width:2px
    classDef decision fill:#fcf,stroke:#333,stroke-width:2px
    classDef local fill:#9cf,stroke:#333,stroke-width:1px
    classDef remote fill:#f99,stroke:#333,stroke-width:1px
    classDef compute fill:#fc9,stroke:#333,stroke-width:1px
    classDef output fill:#cfc,stroke:#333,stroke-width:1px
    classDef final fill:#f96,stroke:#333,stroke-width:2px

    class A start
    class B1 decision
    class C1,D1,E1,F1,G1,H1 local
    class C2,D2,E2,G2,H2 remote
    class I compute
    class J1,J2,J3 output
    class K final
```

## Advanced Memory Optimization Strategy

```mermaid
graph TD
    A[Large Embedding Data] --> B{Size > Memory?}
    B -->|No| C1[Direct Processing]
    B -->|Yes| C2[Memory-Optimized Flow]

    C2 --> D[Batched Processing Strategy]

    D --> E1[Chunking]
    D --> E2[Streaming]
    D --> E3[Checkpointing]

    E1 --> F1[Process in Chunks]
    E1 --> F2[Size: 1000 queries/chunk]

    E2 --> G1[Stream Results]
    E2 --> G2[Progressive UI Updates]

    E3 --> H1[Save Intermediate Results]
    E3 --> H2[Resume Capability]

    F1 --> I[Merge Chunk Results]
    G1 --> I
    H1 --> I

    C1 --> J[Calculate Final Metrics]
    I --> J

    J --> K[Store in Shared State]
    K --> L[Visualization & Report]

    %% Styling
    classDef data fill:#9cf,stroke:#333,stroke-width:2px
    classDef decision fill:#fcf,stroke:#333,stroke-width:2px
    classDef strategy fill:#f99,stroke:#333,stroke-width:2px
    classDef technique fill:#cfc,stroke:#333,stroke-width:1px
    classDef detail fill:#ff9,stroke:#333,stroke-width:1px
    classDef process fill:#fc9,stroke:#333,stroke-width:2px
    classDef output fill:#9f9,stroke:#333,stroke-width:2px

    class A data
    class B decision
    class C1,C2 strategy
    class D strategy
    class E1,E2,E3 technique
    class F1,F2,G1,G2,H1,H2 detail
    class I,J,K process
    class L output
```

## 🔍 Knowledge Graph Visualization

The interactive knowledge graph visualization provides:

- **Node Representation**: Entities in the knowledge base
- **Edge Connections**: Similarity relationships between nodes
- **Color Coding**: Visual differentiation of node types
- **Interactive Exploration**: Zoom, pan, and click for details
- **Filtering**: Focus on specific node types or relationship strengths

## 📊 Sample Reports

The system generates comprehensive reports including:

- **Performance Summary**: Key metrics and performance indicators
- **Detailed Analysis**: Breakdown of performance across query types
- **Comparative Metrics**: Comparison against baseline systems
- **Knowledge Graph Insights**: Network analysis of node relationships
- **Recommendations**: Suggestions for system improvement

## 🧩 Project Structure

```
.
├── .env                  # Environment variables
├── agents/
│   ├── __init__.py
│   ├── main_agent.py     # Main supervisory agent
│   └── stark_agent.py    # StarkQA evaluation agent
├── config/
│   └── config.py         # Configuration settings
├── state/
│   └── shared_state.py   # Shared state management
├── tests/
│   ├── __init__.py
│   └── test_stark_evaluation.py
├── tools/
│   ├── __init__.py
│   └── stark/
│       ├── __init__.py
│       └── evaluation_retrival.py  # Core evaluation tool
├── ui/
│   ├── stark_app.py            # Streamlit interface
│   └── demo_app.py
└── utils/
    ├── __init__.py
    └── llm.py            # LLM utilities
```

## 🔬 Why STaRK?

STaRK addresses critical challenges in modern retrieval systems:

1. **Semi-structured Data**: Real-world knowledge bases combine structured and unstructured data
2. **Complex Queries**: Users formulate queries involving multiple relationships and text
3. **Domain Diversity**: Different domains require specialized evaluation approaches

### Target Users

- **ML Engineers**: Evaluating retrieval system performance
- **Researchers**: Benchmarking LLM capabilities on complex knowledge bases
- **Product Teams**: Optimizing search and recommendation systems
- **Domain Experts**: Analyzing domain-specific retrieval effectiveness

### Real-world Applications

- **E-commerce**: Improved product search leveraging reviews and specifications
- **Academic Research**: Enhanced literature search across citation networks
- **Healthcare**: Better drug discovery through relationship analysis
- **Enterprise Search**: More effective information retrieval from corporate knowledge bases

## Agent Communication Pattern

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant MA as Main Agent
    participant SQA as StarkQA Agent
    participant ET as Evaluation Tool
    participant SS as Shared State

    User->>UI: Upload Files & Configure
    UI->>MA: Request Evaluation
    MA->>SQA: Route Evaluation Request
    SQA->>ET: Invoke Tool

    ET->>ET: Process Embeddings
    Note over ET: Memory-optimized processing

    ET->>SS: Store Results
    ET->>SQA: Return Results
    SQA->>MA: Update Status
    MA->>UI: Display Results
    UI->>User: Show Visualizations

    User->>UI: Ask Question
    UI->>MA: Process Query
    MA->>SQA: Request Context
    SQA->>SS: Retrieve Data
    SS->>SQA: Return Context
    SQA->>MA: Provide Answer
    MA->>UI: Display Answer
    UI->>User: Show Response
```

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- STaRK benchmark creators (Stanford SNAP Group)
- LangChain and LangGraph for agent framework
- Streamlit for the UI framework
- PyTorch and TorchMetrics for evaluation metrics
- OpenAI for GPT-4-mini access
- This project was developed for Team VPE during the Biodatathon - [VirtualPatientEngine/AIAgents4Pharma](https://github.com/VirtualPatientEngine/AIAgents4Pharma)

````markdown
## 📚 Citation

If you use the STaRK Benchmark Suite in your research or project, please cite:

```bibtex
@software{stark_benchmark,
  author = {Kumar, Ansh and Apoorva Gupta},
  title = {STaRK: Benchmarking LLM Retrieval on Semi-structured Knowledge Bases},
  url = {https://github.com/ansh-info/stark-agent},
  year = {2025},
  month = {February},
  note = {STaRK Agent: A hierarchical agent system for evaluating LLM retrieval performance}
}
```
````

## 📞 Contact

For questions or feedback, please open an issue or contact us at [ansh-info@example.com]
