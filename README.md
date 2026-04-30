# JobFit-Agent: AI Resume Matcher & Agentic Improver

JobFit-Agent is a sophisticated system that evaluates how well a resume matches a job description using a fine-tuned Longformer model and autonomously improves LaTeX resumes using an Agentic AI workflow.

## Features

- **ML Match Score**: Uses `allenai/longformer-base-4096` fine-tuned on 1k+ real-world resume/JD pairs for regression scoring.
- **Agentic AI Improver**: An autonomous loop that analyzes gaps, modifies LaTeX code, and verifies factuality using LLMs (Gemini, OpenAI, etc.).
- **Premium Dashboard**: A React + Tailwind CSS frontend with interactive gauges and live improvement logs.
- **Switchable LLM Providers**: Support for Google Gemini, OpenAI, Anthropic, and more via a flexible adapter pattern.

## Setup

### Prerequisites

- Python 3.10+
- Node.js & npm
- NVIDIA GPU (RTX 3060 or better recommended for training)
- LaTeX distribution (MiKTeX or TeX Live) for local compilation (optional, required for full PDF generation)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NischalGautam8/JobFit-Agent.git
   cd resume-reviewer
   ```

2. **Backend Setup**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**:
   ```bash
   cd frontend
   npm install
   ```

### Usage

1. **Data Preparation**:
   ```bash
   python data/prepare_dataset.py
   ```

2. **Training the Model**:
   ```bash
   python training/train.py
   ```

3. **Running the API**:
   ```bash
   python -m uvicorn api.main:app --reload
   ```

4. **Running the Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

## Project Structure

- `data/`: Data loading and preparation scripts.
- `training/`: Model architecture and training loops.
- `api/`: FastAPI backend implementation.
- `agent/`: Agentic AI logic and LLM adapters.
- `frontend/`: React frontend source code.
- `models/`: Trained model checkpoints.

## License

MIT
