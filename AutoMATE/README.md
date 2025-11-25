# AutoMATE - Automotive Multimodal Augmented Technical Expert

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A state-of-the-art **Retrieval-Augmented Generation (RAG)** system for processing and querying automotive technical manuals. AutoMATE combines advanced OCR, multimodal embeddings, and large language models to provide accurate, context-aware answers from vehicle documentation.

![AutoMATE Architecture](docs/architecture.png)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation Framework](#evaluation-framework)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

- **Multimodal RAG Pipeline**: Processes both text and images from PDF manuals
- **Two-Stage Retrieval**: Combines text similarity (Stage 1) with CLIP visual re-ranking (Stage 2)
- **Vehicle-Aware Filtering**: Automatic Named Entity Recognition for vehicle models
- **Cross-Language Support**: Query in English, retrieve from Italian manuals, respond in target language
- **Smart Image Quality Filtering**: Configurable thresholds to exclude low-quality images

### Advanced Features

- **LLM-as-Judge Evaluation**: Automated quality assessment using Gemini
- **Multi-Source Confidence Scoring**: Aggregated confidence from retrieval, relevance, and answer quality
- **Query Augmentation**: Automatic keyword expansion for improved image retrieval
- **Prompt Injection Protection**: Security layer against adversarial queries
- **Auto-Updating Reports**: Markdown evaluation reports generated after each query

### Supported Vehicles

| Vehicle           | Manual ID        |
| ----------------- | ---------------- |
| Fiat 500          | `500`          |
| Fiat Panda        | `PANDA`        |
| Fiat Grande Punto | `GRANDE-PUNTO` |
| Peugeot 208       | `PEUGEOT 208`  |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                │
│              (Italian / English / Auto-detect)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Security   │  │   Vehicle    │  │  Language Detection  │   │
│  │  Sanitizer   │  │  Detector    │  │  + Query Translation │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│      TEXT RETRIEVAL     │     │     IMAGE RETRIEVAL     │
│  ┌───────────────────┐  │     │  ┌───────────────────┐  │
│  │  ChromaDB + E5    │  │     │  │ Stage 1: Text Sim │  │
│  │  (Multilingual)   │  │     │  └───────────────────┘  │
│  └───────────────────┘  │     │           │             │
│           │             │     │           ▼             │
│           ▼             │     │  ┌───────────────────┐  │
│  ┌───────────────────┐  │     │  │ Stage 2: CLIP     │  │
│  │ Vehicle Filtering │  │     │  │ Re-ranking        │  │
│  └───────────────────┘  │     │  └───────────────────┘  │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GENERATION LAYER                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Gemini 2.0 Flash Lite                        │   │
│  │         (Cross-language aware prompts)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Confidence │  │ LLM Judge  │  │ Retrieval  │  │ Auto     │  │
│  │ Calculator │  │ (optional) │  │ Metrics    │  │ Report   │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Pipeline Components

| Component                  | Description                               | Technology                         |
| -------------------------- | ----------------------------------------- | ---------------------------------- |
| **OCR Engine**       | Extracts text and images from PDF manuals | Dolphin 1.5 (ByteDance)            |
| **Text Embeddings**  | Multilingual semantic representations     | `intfloat/multilingual-e5-large` |
| **Image Embeddings** | Visual-semantic alignment                 | `openai/clip-vit-base-patch32`   |
| **Vector Store**     | Persistent similarity search              | ChromaDB                           |
| **LLM**              | Response generation                       | Gemini 2.0 Flash Lite              |
| **Orchestration**    | RAG pipeline management                   | LangChain                          |
| **Web Interface**    | Interactive demo                          | Streamlit + ngrok                  |

---

## Tech Stack

### Core Dependencies

```
langchain>=0.2.0
langchain-chroma>=0.1.0
langchain-google-genai>=1.0.0
langchain-huggingface>=0.0.3
chromadb>=0.4.0
sentence-transformers>=2.2.0
transformers>=4.35.0
```

### ML/AI Models

| Model                 | Purpose          | Size   |
| --------------------- | ---------------- | ------ |
| Dolphin 1.5           | Document OCR     | ~2GB   |
| multilingual-e5-large | Text embeddings  | ~1.1GB |
| CLIP ViT-B/32         | Image embeddings | ~600MB |
| Gemini 2.0 Flash Lite | Generation       | API    |

### Infrastructure

- **Runtime**: Google Colab (T4/V100 GPU)
- **Storage**: Google Drive (persistent)
- **Deployment**: Streamlit + ngrok tunnel

---

## Installation

### Prerequisites

- Google Colab account with GPU runtime
- Google Drive with ~10GB free space
- Ngrok account (free tier)
- Google AI Studio API key

### Quick Start

1. **Open in Colab**

   Upload the notebook to Google Colab or click the badge above.
2. **Configure API Keys**

```python
   GOOGLE_API_KEY = "your-gemini-api-key"
   NGROK_AUTH_TOKEN = "your-ngrok-token"
```

3. **Run Setup (Block 0)**

   Installs dependencies, clones Dolphin OCR, downloads models.
4. **Configure (Block 1)**

   Sets paths, model names, and quality thresholds.
5. **Ingest Documents (Optional)**

   Process new PDF manuals into vector databases.
6. **Launch App (Block 4)**

   Starts Streamlit interface with ngrok tunnel.

---

## Project Structure

```
AutoMATE/
├── notebooks/
│   └── AutoMATE_Complete.ipynb    # Main Colab notebook
├── docs/
│   ├── architecture.png           # System architecture diagram
│   └── evaluation_report.md       # Sample evaluation report
├── configs/
│   └── default_config.py          # Configuration template
├── evaluation/
│   ├── logs/
│   │   └── queries.jsonl          # Query history (JSONL)
│   └── reports/
│       └── automate_evaluation_report.md
├── README.md
├── LICENSE
└── requirements.txt
```

### Notebook Blocks

| Block | Name                 | Description                                 |
| ----- | -------------------- | ------------------------------------------- |
| 0     | Setup                | Dependencies, Dolphin OCR, model downloads  |
| 1     | Configuration        | Centralized config (Single Source of Truth) |
| 1.5   | Cleanup              | Pre-ingestion database cleanup (optional)   |
| 2     | Evaluation Framework | Confidence calculator, LLM Judge, metrics   |
| 2.5   | Advanced RAG         | Vehicle detector, quality filter, security  |
| 3     | Test RAG             | Single query testing with visualization     |
| 4     | Streamlit App        | Full web interface with all features        |

---

## Usage

### Web Interface

1. Launch Block 4 and access the ngrok URL
2. Select response language (Italian/English/Auto)
3. Enter your question about a vehicle
4. View response with confidence scores and relevant images

### Programmatic Usage

```python
# After running Blocks 0-2.5

# Detect vehicle from query
vehicle = VehicleDetector.detect("How does the TRIP button work on Fiat 500?")
# Returns: "500"

# Retrieve relevant chunks
docs = text_db.similarity_search_with_score(
    query, k=6, filter={"manual": vehicle}
)

# Calculate confidence
confidence = ConfidenceCalculator.aggregate(
    retrieval_conf, relevance_conf, quality_conf
)
```

### Query Examples

| Language | Query                                   | Expected Behavior                      |
| -------- | --------------------------------------- | -------------------------------------- |
| Italian  | "Come funziona il pulsante TRIP?"       | Direct retrieval, Italian response     |
| English  | "How do I reset the service indicator?" | Translated retrieval, English response |
| Mixed    | "What is the funzione Start&Stop?"      | Auto-detect, respond in query language |

---

## Configuration

### Central Configuration (Block 1)

All parameters are defined in a single location for consistency:

```python
# Paths
PDF_INPUT_DIR = "/content/drive/MyDrive/MANUALS"
CHROMA_TEXT_DIR = "/content/drive/MyDrive/OCR/Testo_DB"
CHROMA_IMAGE_DIR = "/content/drive/MyDrive/OCR/Images_DB"

# Models
TEXT_EMBED_MODEL = "intfloat/multilingual-e5-large"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Retrieval
STAGE1_TOP_K = 40    # Text similarity candidates
STAGE2_TOP_K = 5     # Final results after CLIP re-ranking
CLIP_WEIGHT = 0.40   # Balance between CLIP and text similarity

# Image Quality Thresholds
IMAGE_QUALITY_CONFIG = {
    'min_width': 20,
    'min_height': 20,
    'min_size_kb': 0.3,
    'max_aspect_ratio': 12.0
}
```

### Tuning Recommendations

| Scenario                   | Parameter        | Suggested Change    |
| -------------------------- | ---------------- | ------------------- |
| Missing relevant chunks    | `STAGE1_TOP_K` | Increase to 60      |
| Too many irrelevant images | `CLIP_WEIGHT`  | Increase to 0.5-0.6 |
| Cross-language issues      | `CHUNK_SIZE`   | Increase to 1500    |
| Slow performance           | `STAGE2_TOP_K` | Decrease to 3       |

---

## Evaluation Framework

### Confidence Scoring (Multi-Source)

AutoMATE calculates confidence from three independent sources:

| Source              | Weight | Description                           |
| ------------------- | ------ | ------------------------------------- |
| **Retrieval** | 40%    | Chunk similarity + consistency        |
| **Relevance** | 35%    | Query-context semantic alignment      |
| **Quality**   | 25%    | Response structure + technical detail |

### Confidence Levels

| Level            | Score Range | Interpretation                             |
| ---------------- | ----------- | ------------------------------------------ |
| **HIGH**   | ≥ 0.75     | System confident, answer reliable          |
| **MEDIUM** | 0.55 - 0.75 | Reasonable confidence, verify if critical  |
| **LOW**    | < 0.55      | Uncertain, manual verification recommended |

### LLM-as-Judge

Optional evaluation using Gemini to score responses on:

- **Faithfulness** (1-5): Alignment with source context
- **Relevance** (1-5): Direct answer to user question
- **Completeness** (1-5): Technical detail and thoroughness

### Auto-Generated Reports

After each query, the system updates a Markdown report with:

- Aggregate statistics (mean, std, range)
- Last 10 queries with full metrics
- Performance breakdowns (RAG time, LLM time, image time)

Report location: `/content/drive/MyDrive/OCR/evaluation/reports/automate_evaluation_report.md`

---

## API Reference

### Core Classes

#### `VehicleDetector`

```python
@classmethod
def detect(cls, query: str) -> Optional[str]:
    """
    Detect vehicle model from query text.
  
    Args:
        query: User query string
      
    Returns:
        Manual identifier (e.g., "500", "PANDA") or None
    """
```

#### `ConfidenceCalculator`

```python
@staticmethod
def aggregate(retrieval: float, relevance: float, quality: float) -> Dict:
    """
    Combine scores into final confidence.
  
    Returns:
        {
            'score': float,      # 0.0 - 1.0
            'label': str,        # "HIGH", "MEDIUM", "LOW"
            'color': str,        # Hex color for UI
            'breakdown': Dict    # Individual scores
        }
    """
```

#### `ImageQualityFilter`

```python
@staticmethod
def filter_results(
    img_results: List,
    threshold: ImageQualityThreshold,
    enabled: bool = True
) -> List:
    """
    Filter images by quality thresholds.
  
    Args:
        img_results: List of (Document, score, clip_used) tuples
        threshold: Quality threshold configuration
        enabled: Whether to apply filtering
      
    Returns:
        Filtered list of results
    """
```

#### `LLMJudge`

```python
def evaluate_response(
    self,
    query: str,
    context: str,
    response: str
) -> Dict:
    """
    Evaluate RAG response quality using LLM.
  
    Returns:
        {
            'faithfulness': int,   # 1-5
            'relevance': int,      # 1-5
            'completeness': int,   # 1-5
            'average': float,      # Mean score
            'reasoning': str       # Brief explanation
        }
    """
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all public functions
- Update README for new features
- Test on Colab before submitting

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [ByteDance Dolphin](https://github.com/ByteDance/Dolphin) - Document OCR
- [LangChain](https://github.com/langchain-ai/langchain) - RAG orchestration
- [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- [OpenAI CLIP](https://github.com/openai/CLIP) - Visual embeddings
- [Google Gemini](https://ai.google.dev/) - Language model

---

## Citation

If you use AutoMATE in your research, please cite:

```bibtex
@software{automate2025,
  author = {Your Name},
  title = {AutoMATE: Automotive Multimodal Augmented Technical Expert},
  year = {2025},
  url = {https://github.com/yourusername/AutoMATE}
}
```

---

<p align="center">
  <b>AutoMATE</b> - Making automotive manuals accessible through AI
</p>
