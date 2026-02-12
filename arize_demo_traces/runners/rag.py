"""Backward-compatible re-export. Actual implementation moved to langchain/rag.py."""
from .langchain.rag import run_rag

__all__ = ["run_rag"]
