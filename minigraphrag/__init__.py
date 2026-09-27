"""MiniGraphRAG — a minimal, complete GraphRAG implementation."""

from .pipeline import MiniGraphRAG
from .answer import Answer, Citation
from .llm import LLMClient, OpenAICompatibleLLM, FakeLLM

__all__ = [
    "MiniGraphRAG",
    "Answer",
    "Citation",
    "LLMClient",
    "OpenAICompatibleLLM",
    "FakeLLM",
]
__version__ = "0.1.0"
