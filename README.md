# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, a local open-source LLM served by [Ollama](https://ollama.com) for AI generation, and provides a web interface for interaction. The entire stack runs locally — **no external API keys are required**.

## Features

- **Fully local & private** — LLM (Ollama) and embeddings (SentenceTransformers) both run on your machine; no data leaves it.
- **Tool-based retrieval** — the model uses search tools to look up course content and outlines, supporting up to 2 sequential rounds of tool calls per query.
- **PDF & TXT ingestion** — documents in `docs/` (including nested subfolders) are loaded and chunked on startup.
- **Source attribution** — answers cite the courses/lessons they came from, with lesson links where available.
- **Session memory** — conversation history is kept per session for follow-up questions.

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- [Ollama](https://ollama.com) installed and running, with a **tool-capable** model pulled:
  ```bash
  ollama pull llama3.1
  ```
  > ⚠️ The model **must** support tool calling, or course queries will fail. Verify with `ollama show <model>` (look for `tools` under Capabilities). Models like `gemma2:2b` do **not** support tools and won't work. Good alternatives: `qwen3`, `llama3.2`, `mistral-nemo`.

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **(Optional) Configure the model**

   Defaults work out of the box for a local Ollama install. To override, create a
   `.env` file in the root directory:
   ```bash
   OLLAMA_BASE_URL=http://localhost:11434/v1
   OLLAMA_MODEL=llama3.1
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Adding Course Materials

Place `.pdf` or `.txt` files in the `docs/` directory (subfolders are scanned recursively) — they're ingested automatically on startup. Already-ingested courses (matched by title) are skipped, so restarts are cheap.

For best results, start each document with this header so metadata and lessons are extracted correctly:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 1: <lesson title>
Lesson Link: <url>
<lesson content...>
```
Files without this header still ingest, but are treated as a single untitled block.

> Note: scanned/image-only PDFs have no extractable text layer and won't be indexed (there is no OCR). PDFs with a real text layer work fine.

## Running Tests

```bash
uv run pytest                 # full suite
uv run pytest -m unit         # by marker: unit | integration | api | slow
```

