import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG system"""

    # LLM settings (Ollama via its OpenAI-compatible endpoint)
    # No API key is needed for local Ollama; the OpenAI client just requires a
    # non-empty string, so OLLAMA_API_KEY defaults to a dummy value.
    # default_factory reads env at instantiation time (after load_dotenv), so a
    # .env file is respected and overrides work without re-importing the module.
    OLLAMA_BASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
    )
    OLLAMA_MODEL: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1")
    )
    OLLAMA_API_KEY: str = field(
        default_factory=lambda: os.getenv("OLLAMA_API_KEY", "ollama")
    )

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800  # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100  # Characters to overlap between chunks
    MAX_RESULTS: int = 5  # Maximum search results to return
    MAX_HISTORY: int = 2  # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location


config = Config()
