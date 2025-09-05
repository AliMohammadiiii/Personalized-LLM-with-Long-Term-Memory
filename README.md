# Personalized LLM Assistant with Long-Term Memory

This repository accompanies the paper *"Personalized Large-Language-Model Assistant with Long-Term User Memory via Retrieval-Augmented Generation"*. It provides a minimal, modular implementation of the system described in the paper.

## Repository Structure

- `src/llm_assistant/` – core Python package
  - `memory.py` – FAISS-backed memory module
  - `llm_client.py` – selects between OpenAI, Hugging Face, or Google models
  - `dialogue.py` – retrieval‑augmented generator and dialogue manager
- `src/main.py` – console entry point using the dialogue manager
- `notebooks/` – Jupyter notebooks used during development
- `docs/` – project report and additional documentation

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
# edit .env and choose an `LLM_PROVIDER` and API keys
```

## Usage

Run the interactive console:

```bash
python src/main.py
```

Type questions at the prompt; the assistant retrieves relevant memories, generates a response with the chosen LLM provider, and learns new personal facts from your inputs.

To reproduce the experiments and explore additional usage examples, open the Jupyter notebook at `notebooks/LLMProject.ipynb`.

## Citation

If you use this codebase, please cite the accompanying paper.
