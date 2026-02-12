"""Backward-compatible re-export. Actual implementation moved to langchain/text_to_sql.py."""
from .langchain.text_to_sql import run_text_to_sql

__all__ = ["run_text_to_sql"]
