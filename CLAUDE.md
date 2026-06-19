# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start using shell script
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Add new dependencies (use uv instead of pip)
uv add package_name

# LLM runs locally via Ollama (no API key needed). Ensure Ollama is running and
# the model is pulled:
ollama pull llama3.1

# Optional overrides via .env (defaults shown):
# OLLAMA_BASE_URL=http://localhost:11434/v1
# OLLAMA_MODEL=llama3.1
```

### Python Execution
Always use `uv` for running Python files and commands:
```bash
# Run Python scripts
uv run python script.py

# Run any Python command
uv run command_name
```

### Running Tests
```bash
# Run full test suite (pytest configured in pyproject.toml)
uv run pytest

# Run by marker (unit | integration | api | slow)
uv run pytest -m unit
```
Tests live in `backend/tests/` with shared fixtures in `conftest.py`.

### Code Quality Tools

#### Prerequisites
Install development dependencies before using code quality tools:
```bash
uv sync --group dev
```

#### Available Scripts

**Format Script (Modifies Files)**
```bash
./scripts/format.sh
```
Use this script when you want to automatically fix code style issues. It will:
1. Sort imports with isort
2. Format code with Black 
3. Run flake8 linting (reports remaining issues)
4. Run mypy type checking

**Lint Script (Read-Only Checks)**
```bash
./scripts/lint.sh
```
Use this script to verify code quality without modifying files. Perfect for:
- Pre-commit checks
- CI/CD pipelines
- Verifying code before submitting PRs

Exit code 0 = all checks pass, non-zero = issues found.

#### Troubleshooting
If scripts aren't executable: `chmod +x scripts/*.sh`

### Application Access
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials with a FastAPI backend and vanilla JavaScript frontend.

### Core Components

**RAGSystem (backend/rag_system.py)**: Main orchestrator that coordinates all components
- Manages document processing, vector storage, AI generation, and search tools
- Handles course document ingestion from the docs/ directory
- Processes queries using tool-based search approach

**VectorStore (backend/vector_store.py)**: ChromaDB-based vector storage with dual collections
- `course_catalog`: Stores course titles for name resolution
  - Metadata: title, instructor, course_link, lesson_count, lessons_json (list of lessons with lesson_number, lesson_title, lesson_link)
- `course_content`: Stores text chunks for semantic search
  - Metadata: course_title, lesson_number, chunk_index
- Supports filtered search by course name and lesson number

**AIGenerator (backend/ai_generator.py)**: Local LLM integration via Ollama
- Uses the OpenAI SDK pointed at Ollama's OpenAI-compatible endpoint (`OLLAMA_BASE_URL`); default model `llama3.1` (`OLLAMA_MODEL`)
- Converts the tools' Anthropic-style `input_schema` definitions to OpenAI function format at call time
- Implements tool calling for search functionality
- Supports up to 2 sequential rounds of tool calls per query (gather then refine)
- Maintains conversation history via SessionManager

**Search Tools (backend/search_tools.py)**: Tool-based search system
- CourseSearchTool: Semantic search across course content with intelligent course name resolution
- CourseOutlineTool: Returns course structure (title, link, full lesson list) for overview-style queries
- ToolManager: Manages tool registration and execution for AI model

### Data Flow
1. Course documents (PDF, TXT) are loaded recursively from docs/ (and subfolders) on startup
2. DocumentProcessor reads files (PDFs via pypdf), chunks content, and extracts course metadata
3. VectorStore stores both metadata and content in separate ChromaDB collections
4. User queries trigger AI generation with access to search tools
5. AI uses CourseSearchTool to find relevant content and generates responses
6. Frontend displays responses with source attribution

### Key Configuration (backend/config.py)
- Chunk size: 800 characters with 100 character overlap
- Embedding model: all-MiniLM-L6-v2 (SentenceTransformers)
- Max search results: 5 per query
- Conversation history: 2 message pairs

### Frontend Architecture
- Single-page application with vanilla JavaScript
- Real-time course statistics display
- Markdown rendering support for AI responses
- Responsive design with sidebar for course info and suggested queries

## Development Notes

- The system automatically loads documents from docs/ (recursively, incl. subfolders) on startup; already-ingested courses are skipped by title
- Only `.pdf` and `.txt` are ingested. Scanned/image-only PDFs have no text layer and won't be indexed (no OCR)
- The Ollama model **must support tool calling** or course queries fail. Check with `ollama show <model>` (look for `tools`). `gemma2:2b` does NOT support tools; `qwen3`/`llama3.2`/`mistral-nemo` are viable alternatives
- ChromaDB data persists in backend/chroma_db/ (created on first run; ~2-5 MB for the current docs)
- FastAPI serves both API endpoints (/api/*) and static frontend files
- CORS is configured for development with broad permissions
- No-cache headers are set for static files during development
- This machine's `uv` requires `--link-mode=copy` (hardlink error otherwise), e.g. `uv run --link-mode=copy pytest`